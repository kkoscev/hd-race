# HD-RACE: Spray-based Local Tone Mapping Operator
Reference code for the paper [HD-RACE: Spray-based Local Tone Mapping Operator](https://hal.archives-ouvertes.fr/hal-03276191/file/HD_RACE___ISPA2021.pdf). Karlo Koščević, Vedran Stipetić, Edoardo Provenzi, Marko Subašić, Sven Lončarić. In ISPA 2021. If you use this code, please cite our paper:
```
@inproceedings{koscevic2021hd,
  author={Ko{\v{s}}{\v{c}}evi{\'c}, Karlo and Stipeti{\'c}, Vedran and Provenzi, Edoardo and Bani{\'c}, Nikola and Suba{\v{s}}i{\'c}, Marko and Lon{\v{c}}ari{\'c}, Sven}
  booktitle={2021 12th International Symposium on Image and Signal Processing and Analysis (ISPA)},
  title={HD-RACE: Spray-based Local Tone Mapping Operator},
  year={2021},
  pages={264-269},
  doi={10.1109/ISPA52656.2021.9552145}}
```

## Abstract
*In this paper, a local tone mapping operator is proposed. It is based on the theory of sprays introduced in the Random Sprays Retinex algorithm, a white balance algorithm dealing with the locality of color perception. This tone mapping implementation compresses high dynamic range images by using three types of computations on sprays. These operations are carefully chosen so that the result of their convex combination is a low dynamic range image with a high level of detail in all its parts regardless of the original luminance values that may span over large dynamic ranges. Furthermore, a simple local formulation of the Naka-Rushton equation based on random sprays is given. The experimental results are presented and discussed.*

## Code

#### Dependencies:
- numpy
- opencv-python

To run the code run hd-race.py script with arguments:
- `hdr_path`: path to the HDR image
- `N`: number of sprays (default 15),
- `n`: number of points in a spray (default 250),
- `guided_filter`: specifies to use [Guided Image Filter](),
- `k`: window size for guieded filter (default 25),
- `output_dir`: path to save output image.

`python3 hd_race.py <PATH TO THE HDR IMAGE> -N 15 -n 250 --guided_filter --output_path <PATH TO OUTPUT DIRECTORY>`

## Results
