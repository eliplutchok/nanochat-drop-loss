"""
Python implementation of the Rust BPE tokenizer (lib.rs)
This is a direct translation to help understand the algorithm.
"""

import heapq
import logging
import regex
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from multiprocessing import Pool, cpu_count

# Default GPT-4 style regex pattern for splitting text
GPT4_PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

Pair = Tuple[int, int]

# ------------------------ internal helpers ------------------------

class Word:
    """Represents a sequence of token IDs"""
    
    def __init__(self, ids: List[int]):
        self.ids = ids
    
    def pairs(self):
        """Iterator over adjacent pairs in this word"""
        for i in range(len(self.ids) - 1):
            yield (self.ids[i], self.ids[i + 1])
    
    def merge_pair(self, pair: Pair, new_id: int) -> List[Tuple[Pair, int]]:
        """
        Merge all non-overlapping occurrences of pair -> new_id.
        Returns a list of local pair-count deltas for THIS word only:
          -1 for removed pairs, +1 for newly created pairs.
        
        NOTE: this version deliberately avoids a HashMap in the hot loop.
        """
        a, b = pair
        n = len(self.ids)
        if n < 2:
            return []
        
        out: List[int] = []
        deltas: List[Tuple[Pair, int]] = []
        
        i = 0
        while i < n:
            if i + 1 < n and self.ids[i] == a and self.ids[i + 1] == b:
                left = out[-1] if out else None
                right = self.ids[i + 2] if i + 2 < n else None
                
                # remove old pairs
                if left is not None:
                    deltas.append(((left, a), -1))
                    deltas.append(((left, new_id), 1))
                deltas.append(((a, b), -1))
                if right is not None:
                    deltas.append(((b, right), -1))
                    deltas.append(((new_id, right), 1))
                
                # write merged token
                out.append(new_id)
                i += 2  # skip 'a' and 'b'
            else:
                out.append(self.ids[i])
                i += 1
        
        self.ids = out
        return deltas


class MergeJob:
    """
    Represents a merge candidate with its frequency count and affected positions.
    Used in a max-heap (by count, tie-breaking by pair order for determinism).
    """
    
    def __init__(self, pair: Pair, count: int, pos: Set[int]):
        self.pair = pair
        self.count = count
        self.pos = pos
    
    def __lt__(self, other):
        # Max-heap by count; tie-break to ascending pair order (deterministic)
        # Note: Python's heapq is a min-heap, so we negate for max-heap behavior
        if self.count != other.count:
            return self.count > other.count  # reverse for max-heap
        else:
            # ascending order on the pair when counts tie
            return self.pair < other.pair
    
    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair


def count_pairs_for_word(args) -> Tuple[Dict[Pair, int], Dict[Pair, Set[int]]]:
    """Helper function for parallel pair counting"""
    i, word, count = args
    local_pc: Dict[Pair, int] = defaultdict(int)
    local_wtu: Dict[Pair, Set[int]] = defaultdict(set)
    
    if len(word.ids) >= 2 and count != 0:
        for pair in word.pairs():
            local_pc[pair] += count
            local_wtu[pair].add(i)
    
    return (local_pc, local_wtu)


def count_pairs_parallel(words: List[Word], counts: List[int]) -> Tuple[Dict[Pair, int], Dict[Pair, Set[int]]]:
    """
    Count all pairs across all words in parallel.
    Returns:
      - pair_counts: frequency of each pair
      - where_to_update: which word indices contain each pair
    """
    # Prepare args for parallel processing
    args_list = [(i, w, counts[i]) for i, w in enumerate(words)]
    
    # Process in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(count_pairs_for_word, args_list)
    
    # Merge results
    pair_counts: Dict[Pair, int] = defaultdict(int)
    where_to_update: Dict[Pair, Set[int]] = defaultdict(set)
    
    for local_pc, local_wtu in results:
        for k, v in local_pc.items():
            pair_counts[k] += v
        for k, s in local_wtu.items():
            where_to_update[k].update(s)
    
    return (dict(pair_counts), dict(where_to_update))


# ------------------------ END helpers ------------------------

class Tokenizer:
    """A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation"""
    
    def __init__(self):
        # Maps pairs of token IDs to their merged token ID
        self.merges: Dict[Pair, int] = {}
        # The regex pattern used for text splitting
        self.pattern: str = ""
        # Compiled regex for efficiency
        self.compiled_pattern = None
    
    def train_core_incremental(self, words: List[Word], counts: List[int], vocab_size: int):
        """
        Core incremental BPE training given unique words and their counts.
        `words`: one entry per unique chunk (List[int] of token-ids/bytes).
        `counts`: same length as `words`, count per chunk.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256
        logging.info(f"Starting BPE training: {num_merges} merges to compute")
        self.merges.clear()
        
        # ---- Initial pair_counts and where_to_update (parallel) ----
        logging.info(f"Computing initial pair counts from {len(words)} unique sequences")
        pair_counts, where_to_update = count_pairs_parallel(words, counts)
        
        # ---- Build heap ----
        logging.info(f"Building heap with {len(pair_counts)} unique pairs")
        heap = []
        for pair, pos in where_to_update.items():
            c = pair_counts.get(pair, 0)
            if c > 0:
                heapq.heappush(heap, MergeJob(pair, c, pos))
        
        # ---- Merge loop ----
        logging.info("Starting merge loop")
        merges_done = 0
        last_log_percent = 0
        
        while merges_done < num_merges:
            if not heap:
                break
            
            top = heapq.heappop(heap)
            
            # Lazy refresh
            current = pair_counts.get(top.pair, 0)
            if top.count != current:
                top.count = current
                if top.count > 0:
                    heapq.heappush(heap, top)
                continue
            if top.count == 0:
                break
            
            # Record merge
            new_id = 256 + merges_done
            self.merges[top.pair] = new_id
            
            # Merge this pair in all words where it occurs
            local_pos_updates: Dict[Pair, Set[int]] = defaultdict(set)
            for word_idx in top.pos:
                # Apply merge to this word and collect pair-count deltas
                changes = words[word_idx].merge_pair(top.pair, new_id)
                # Update global pair counts based on this word's count
                for pair, delta in changes:
                    delta_total = delta * counts[word_idx]
                    if delta_total != 0:
                        pair_counts[pair] = pair_counts.get(pair, 0) + delta_total
                        if delta > 0:
                            local_pos_updates[pair].add(word_idx)
            
            # Add the updated pair counts back to the heap
            for pair, pos in local_pos_updates.items():
                cnt = pair_counts.get(pair, 0)
                if cnt > 0:
                    heapq.heappush(heap, MergeJob(pair, cnt, pos))
            
            merges_done += 1
            
            # Log progress every 1%
            current_percent = (merges_done * 100) // num_merges
            if current_percent > last_log_percent:
                logging.info(
                    f"Progress: {current_percent}% ({merges_done}/{num_merges} merges) - "
                    f"Last merge: {top.pair} -> {new_id} (frequency: {top.count})"
                )
                last_log_percent = current_percent
        
        logging.info(f"Finished training: {merges_done} merges completed")
    
    def train_from_iterator(self, iterator, vocab_size: int, buffer_size: int = 8192, pattern: str = None):
        """
        Train from a streaming iterator (parallel ingestion).
        We fill a buffer, then release to do heavy splitting and counting in parallel.
        """
        # Use provided pattern or default to GPT-4 pattern
        pattern_str = pattern if pattern is not None else GPT4_PATTERN
        
        # Update the stored pattern and compile it
        self.pattern = pattern_str
        self.compiled_pattern = regex.compile(pattern_str)
        
        # Global chunk counts
        counts: Dict[str, int] = defaultdict(int)
        
        logging.info(f"Processing sequences from iterator (buffer_size: {buffer_size})")
        total_sequences = 0
        
        # Helper: process a batch of strings in parallel
        def process_batch(batch: List[str]) -> Dict[str, int]:
            def process_one(text: str) -> Dict[str, int]:
                local_counts = defaultdict(int)
                for match in self.compiled_pattern.finditer(text):
                    piece = match.group(0)
                    local_counts[piece] += 1
                return local_counts
            
            with Pool(cpu_count()) as pool:
                results = pool.map(process_one, batch)
            
            # Merge all local counts
            merged = defaultdict(int)
            for local_counts in results:
                for k, v in local_counts.items():
                    merged[k] += v
            return merged
        
        # Stream ingestion loop
        buf = []
        for text in iterator:
            buf.append(text)
            
            if len(buf) >= buffer_size:
                total_sequences += len(buf)
                local_counts = process_batch(buf)
                
                # Merge into global counts
                for k, v in local_counts.items():
                    counts[k] += v
                
                buf = []
        
        # Process remaining buffer
        if buf:
            total_sequences += len(buf)
            local_counts = process_batch(buf)
            for k, v in local_counts.items():
                counts[k] += v
        
        logging.info(f"Processed {total_sequences} sequences total, {len(counts)} unique")
        
        # Materialize words & counts
        words = []
        cvec = []
        for chunk, c in counts.items():
            words.append(Word([b for b in chunk.encode('utf-8')]))
            cvec.append(c)
        
        self.train_core_incremental(words, cvec, vocab_size)
    
    def get_pattern(self) -> str:
        """Return the regex pattern"""
        return self.pattern
    
    def get_mergeable_ranks(self) -> List[Tuple[bytes, int]]:
        """
        Return the mergeable ranks (token bytes -> token id / rank)
        """
        mergeable_ranks = []
        
        # Build vocabulary incrementally from low to high token IDs
        token_bytes: List[bytes] = [bytes([i]) for i in range(256)]
        
        for i, tb in enumerate(token_bytes):
            mergeable_ranks.append((tb, i))
        
        # Sort merges by token id (so we can reconstruct bytes progressively)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        
        for pair, merged_id in sorted_merges:
            left, right = pair
            merged = token_bytes[left] + token_bytes[right]
            
            # Extend token_bytes list if needed
            while len(token_bytes) <= merged_id:
                token_bytes.append(b'')
            token_bytes[merged_id] = merged
            
            mergeable_ranks.append((merged, merged_id))
        
        return mergeable_ranks
    
    def encode(self, text: str) -> List[int]:
        """Encode a string into token IDs"""
        if self.compiled_pattern is None:
            raise ValueError("Tokenizer not trained or pattern not set")
        
        all_ids = []
        
        # Split text using the regex pattern
        for match in self.compiled_pattern.finditer(text):
            chunk = match.group(0)
            
            # Convert chunk to bytes then to int IDs
            ids = [b for b in chunk.encode('utf-8')]
            
            # Apply merges iteratively
            while len(ids) >= 2:
                # Find the best pair to merge
                best_pair = None
                best_idx = None
                best_new_id = None
                
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self.merges:
                        new_id = self.merges[pair]
                        if best_pair is None or new_id < best_new_id:
                            best_pair = pair
                            best_idx = i
                            best_new_id = new_id
                
                # If we found a pair to merge, apply it
                if best_pair is not None:
                    ids[best_idx] = best_new_id
                    ids.pop(best_idx + 1)
                else:
                    # No more merges possible
                    break
            
            all_ids.extend(ids)
        
        return all_ids


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simple test
    tokenizer = Tokenizer()
    
    # Create a simple text iterator
    texts = [
        "Hello world!",
        "Hello there!",
        "Hello hello hello",
        "World world world",
    ] * 100  # repeat for better statistics
    
    # Train
    tokenizer.train_from_iterator(iter(texts), vocab_size=300, buffer_size=100)
    
    # Test encoding
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    print(f"Encoded '{test_text}': {encoded}")
    
    # Show some merges
    print(f"\nLearned {len(tokenizer.merges)} merges")
    print("First 10 merges:")
    sorted_merges = sorted(tokenizer.merges.items(), key=lambda x: x[1])
    for pair, token_id in sorted_merges[:10]:
        print(f"  {pair} -> {token_id}")

