# Third-party provenance

| Component | Source / fixed version | License location |
| --- | --- | --- |
| Mortal / libriichi | Equim-chan/Mortal, 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2 | AGPL-3.0-or-later; `third_party/Mortal/LICENSE` |
| Community ONNX model | hhim8826/mortal3-b40c192, 7784a9deddf63df03f023474116f715850d3c7a0 | model card declares AGPL-3.0; `models/README.md`, `models/LICENSE.upstream` |
| mjai-reviewer / convlog / export test fixtures | 2dc5ec5c8b28517cfb45f57eb21536d9a8f67aa9 | Apache-2.0; `third_party/mjai-reviewer/LICENSE` |
| ONNX Runtime | 1.23.2 | MIT; installed distribution metadata |
| NumPy / psutil / ONNX | versions in requirements.lock | licenses in installed distribution metadata |
| Rust toolchain | 1.90.0 | upstream MIT/Apache and bundled third-party licenses |
| NVIDIA CUDA / cuDNN wheels | exact versions in requirements.lock | NVIDIA license files in installed wheels |
| Windows SDK libraries | Microsoft.Windows.SDK.CPP.x64 10.0.26100.9169 | license material included in `.tools/windows-sdk` package |
| OpenCV, MSS, Pillow | 4.11.0.86, 10.1.0, 11.3.0 | installed distribution metadata; Apache-2.0, MIT, HPND respectively |
| MahjongRepository/mahjong | 1.3.0 | MIT; installed distribution metadata |
| PP-OCRv4 text recognition model distributed by RapidOCR | rapidocr-onnxruntime 1.4.4 wheel, SHA256 recorded in `assets/ocr/manifest.json` | Apache-2.0; `assets/ocr/LICENSE.upstream`; original source https://github.com/RapidAI/RapidOCR/tree/v1.4.4 |
| pjura ViT tile classifier | `pjura/mahjong_soul_vision`, revision `ea01ca0bcb9be0fc10a5fa09ee8356770c25adc5`; original weight SHA256 `a3cc566c8011b4496239a5228de3985e37ba2c2dfceefd52f67bb34c5f7e49c0` | Model card declares Apache-2.0; original card in `models/pjura-vision/upstream/README.md`, provenance and conversion record in `models/pjura-vision/manifest.json`, standard license text in `models/pjura-vision/LICENSE-2.0.txt` |
| Local Mahjong Soul calibration crops and recordings | user-owned Steam client and supplied replay | Game assets remain the property of their owners; local development samples have no redistribution grant |

Preserve upstream sources, license files and model provenance when redistributing. Mortal and the decision model's AGPL obligations apply; this file does not replace their license terms. Runtime libraries are installed locally rather than incorporated into source files. The pjura PyTorch weights are retained only as verified export inputs; PyTorch and Transformers are installed exclusively in `.tools/pjura-export`. The application environment continues to use ONNX Runtime GPU without PyTorch. No opponent-hand prediction network is used. The gated `pjura/mahjong_vision` assistant and its discard model were not downloaded.

Local Mortal modification: `patches/mortal-snapshot` adds a visible current-snapshot constructor to the pinned libriichi state module. The source overlay and upstream/original hashes are recorded in its manifest and in `configs/build.json`. This derivative source retains Mortal’s AGPL-3.0-or-later license; the original event API and model weights are unchanged.
