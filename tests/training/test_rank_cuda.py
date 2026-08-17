from __future__ import annotations

import unittest

from kimodo.evaluation.rank_cuda import pin_local_cuda_device


class PinLocalCudaDeviceTests(unittest.TestCase):
    def test_uses_local_rank_when_all_gpus_are_visible(self) -> None:
        environ = {"LOCAL_RANK": "3"}
        self.assertEqual(pin_local_cuda_device(environ=environ), "3")
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(environ["TEXT_ENCODER_DEVICE"], "cuda:0")

    def test_indexes_into_existing_visible_list(self) -> None:
        environ = {"LOCAL_RANK": "4", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
        self.assertEqual(pin_local_cuda_device(environ=environ), "4")
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], "4")

    def test_leaves_a_single_visible_gpu_alone(self) -> None:
        environ = {"LOCAL_RANK": "0", "CUDA_VISIBLE_DEVICES": "5"}
        self.assertEqual(pin_local_cuda_device(environ=environ), "5")
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], "5")

    def test_does_not_override_explicit_text_encoder_device(self) -> None:
        environ = {"LOCAL_RANK": "1", "TEXT_ENCODER_DEVICE": "cuda:1"}
        pin_local_cuda_device(environ=environ)
        self.assertEqual(environ["TEXT_ENCODER_DEVICE"], "cuda:1")

    def test_rejects_rank_outside_visible_list(self) -> None:
        with self.assertRaises(ValueError):
            pin_local_cuda_device(
                environ={"LOCAL_RANK": "8", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
            )


if __name__ == "__main__":
    unittest.main()
