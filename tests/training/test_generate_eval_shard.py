from __future__ import annotations

import unittest
from pathlib import Path

from kimodo.evaluation.generate_shards import resolve_generation_shard, select_example_shard


class GenerateEvalShardTests(unittest.TestCase):
    def test_cli_flags_override_env(self) -> None:
        index, count = resolve_generation_shard(
            3,
            16,
            environ={"RANK": "0", "WORLD_SIZE": "8"},
        )
        self.assertEqual((index, count), (3, 16))

    def test_torchrun_env_when_flags_omitted(self) -> None:
        index, count = resolve_generation_shard(
            None,
            None,
            environ={"RANK": "7", "WORLD_SIZE": "16"},
        )
        self.assertEqual((index, count), (7, 16))

    def test_single_process_default(self) -> None:
        self.assertEqual(resolve_generation_shard(None, None, environ={}), (0, 1))

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaises(ValueError):
            resolve_generation_shard(16, 16, environ={})

    def test_select_shard_is_disjoint_and_complete(self) -> None:
        examples = [(Path(f"/src/{i}"), Path(str(i))) for i in range(37)]
        shards = [select_example_shard(examples, index, 16) for index in range(16)]
        flattened = [item for shard in shards for item in shard]
        self.assertEqual(set(flattened), set(examples))
        self.assertEqual(len(flattened), 37)
        self.assertEqual(len(set(flattened)), 37)
        self.assertTrue(all(2 <= len(shard) <= 3 for shard in shards))


if __name__ == "__main__":
    unittest.main()
