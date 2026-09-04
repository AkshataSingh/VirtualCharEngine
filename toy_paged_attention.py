import math

BLOCK_SIZE = 16  # tokens per block

class BlockPool:
    """Manages a pool of fixed-size KV-cache blocks, shared across sequences."""
    def __init__(self, num_blocks, block_size=BLOCK_SIZE):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.total_blocks = num_blocks

    def allocate(self):
        if not self.free_blocks:
            raise RuntimeError("Out of memory: no free blocks available")
        return self.free_blocks.pop()

    def free(self, block_id):
        self.free_blocks.append(block_id)

    @property
    def blocks_in_use(self):
        return self.total_blocks - len(self.free_blocks)


class PagedSequence:
    """A sequence whose KV-cache is a list of block references (a 'page table'),
    not one contiguous buffer — new blocks are allocated only as needed."""
    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_table = []
        self.num_tokens = 0

    def append_token(self):
        if self.num_tokens % self.pool.block_size == 0:
            self.block_table.append(self.pool.allocate())
        self.num_tokens += 1

    def free_all(self):
        for block_id in self.block_table:
            self.pool.free(block_id)
        self.block_table = []


def bytes_per_token(num_layers=24, kv_heads=2, head_dim=64, dtype_bytes=4):
    return 2 * num_layers * kv_heads * head_dim * dtype_bytes  # K + V


def simulate(max_seq_len, actual_lengths, block_size=BLOCK_SIZE):
    per_token = bytes_per_token()
    naive_bytes = max_seq_len * per_token * len(actual_lengths)
    num_blocks_needed = sum(math.ceil(length / block_size) for length in actual_lengths)
    paged_bytes = num_blocks_needed * block_size * per_token
    return naive_bytes, paged_bytes


if __name__ == "__main__":
    MAX_SEQ_LEN = 2048
    actual_lengths = [42, 78, 55, 103, 61, 89, 47, 120]  # realistic response lengths

    naive_bytes, paged_bytes = simulate(MAX_SEQ_LEN, actual_lengths)

    print(f"Sequences: {len(actual_lengths)}, lengths: {actual_lengths}")
    print(f"Max seq len (worst-case allocation): {MAX_SEQ_LEN}")
    print(f"\nNaive (pre-allocate max_seq_len per sequence): {naive_bytes / 1024 / 1024:.2f} MB")
    print(f"Paged (block_size={BLOCK_SIZE}, allocate on demand): {paged_bytes / 1024 / 1024:.2f} MB")
    print(f"Memory saved: {(1 - paged_bytes / naive_bytes) * 100:.1f}%")

    print("\n--- Block manager demo ---")
    pool = BlockPool(num_blocks=200)
    seq = PagedSequence(pool)
    for i in range(50):
        seq.append_token()
    print(f"After 50 tokens: {len(seq.block_table)} blocks allocated ({len(seq.block_table) * BLOCK_SIZE} token capacity, {seq.num_tokens} actually used)")
    print(f"Blocks in use in pool: {pool.blocks_in_use}/{pool.total_blocks}")
    seq.free_all()
    print(f"After freeing: {pool.blocks_in_use}/{pool.total_blocks} blocks in use (returned to pool for reuse)")