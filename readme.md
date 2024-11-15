# MFT: Long-Term Tracking of Every Pixel

## A new [MFTIQ](https://github.com/serycjon/MFTIQ) tracker has been published, check it out!

[Project Page](https://cmp.felk.cvut.cz/~serycjon/MFT/)

Official implementation of the MFT tracker from the paper:

[Michal Neoral](https://scholar.google.com/citations?user=fK9nkmQAAAAJ&hl=en&oi=ao), [Jonáš Šerých](https://cmp.felk.cvut.cz/~serycjon/), [Jiří Matas](https://cmp.felk.cvut.cz/~matas/): "**MFT: Long-Term Tracking of Every Pixel**", WACV 2024

![demo_out](https://cmp.felk.cvut.cz/~serycjon/MFT/visuals/demo_out.gif)

Please cite our paper, if you use any of this.

	@inproceedings{neoral2024mft,
	  title={{MFT}: Long-term tracking of every pixel},
	  author={Neoral, Michal and Serych, Jonas and Matas, Jiri},
	  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
	  pages={6837--6847},
	  year={2024},
	}


## Install

Create and activate a new virtualenv:

    # we have tested with python 3.7.4
    python -m venv venv
    source venv/bin/activate

Then install all the dependencies:

    pip install torch numpy einops tqdm opencv-python scipy Pillow==9 matplotlib ipdb


## Run the demo

Simply running:

    python demo.py

should produce a `demo_out` directory with two visualizations.


## Training

See [train.org](train.org)


## License

The demo video in `demo_in` was extracted from [youtube](https://www.youtube.com/watch?v=ugsJtsO9w1A).

This work is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
The `MFT/RAFT` directory contains a modified version of [RAFT](https://github.com/princeton-vl/RAFT), which is licensed under BSD-3-Clause license.
Our modifications (`OcclusionAndUncertaintyBlock` and its integration in `raft.py`) are licensed again under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).


## Acknowledgments

This work was supported by Toyota Motor Europe,
by the Grant Agency of the Czech Technical University in Prague, grant No. `SGS23/173/OHK3/3T/13`, and
by the Research Center for Informatics project `CZ.02.1.01/0.0/0.0/16_019/0000765` funded by OP VVV.

