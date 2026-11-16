# Seeing the Game: Soccer Player Identification from Still Images

**CSC 481 – Introduction to Image Processing · Muzammil Ali Khan & Sidra Azeem**

Given one still frame from a soccer broadcast, we find every person on the pitch, group them into the two teams, goalkeepers and referees, and read each player's jersey number, or answer **"unreadable"** when the number can't be seen.

Everything is built with classical image processing (color spaces, thresholding, morphology, connected components, clustering). The only learned parts are small classifiers **trained from scratch: no pretrained models** are used anywhere.

![Field and team pipeline](figures/field/pipeline_diagram.png)

## Pipeline

**Field and teams** (`notebooks/01_field_and_teams.ipynb`)

1. **Grass mask:** HSV threshold (hue 35–85), so grass keeps the same hue in sun and shade.
2. **Field region:** opening removes specks, closing rejoins grass cut by pitch lines, and the convex hull gives the field.
3. **Players:** non-grass pixels inside the field, with an opening that removes thin pitch lines.
4. **Player boxes:** connected components filtered by size and shape.
5. **Team colors:** dominant (most common quantized) color of each torso, grouped with K-means (K = 2 for teams, K = 5 to also separate referees and goalkeepers).

**Jersey numbers** (`notebooks/02_jersey_numbers.ipynb`)

1. **Usable images:** torso crop, bicubic upscaling, size and sharpness checks.
2. **Shirt color:** K-means in LAB color space, then a contrast map (ΔE) against the shirt inside the player's silhouette.
3. **Number mask:** Otsu thresholding at several scales, opening and closing, border clearing.
4. **Digits:** connected components with shape rules; touching digits are cut at the thinnest column.
5. **Reading:** three classifiers compared, simplest first: shape features (holes, Hu moments, zoning) with kNN/SVM, HOG + SVM, and a small CNN trained from scratch on synthetic and real digits.
6. **Decision:** below a confidence threshold τ the answer is "unreadable"; the six frames of a player vote.

![Jersey number walkthrough](figures/jersey/walkthrough_1.png)

## Data

Both halves use [SoccerNet](https://www.soccer-net.org/), a public benchmark of professional broadcasts with human labels.

| Task | What we use | Size |
| --- | --- | --- |
| Game State Reconstruction | Full frames, every person boxed with role and team | 60 frames from 6 clips · 802 players, 39 goalkeepers, 75 referees |
| Jersey Number Recognition | Close-up crops of one player per clip | 1,427 train / 1,211 test players, 6 frames each · 28% show no number |

Labels are only used to measure results and to train the digit classifiers. Data is not included in this repository; the notebooks download it.

## Results (preliminary)

**Field and teams** (60 frames)

| Detection recall | Detection precision | Team accuracy (K = 2) | Purity (K = 5) |
| --- | --- | --- | --- |
| 0.68 | 0.83 | 0.92 | 0.90 |

Players shorter than 40 px are almost never found (recall 0.00, vs 0.70 above 80 px).

**Jersey numbers** (1,211 test players; τ chosen on validation, test scored once)

| Method | Player accuracy |
| --- | --- |
| Always "unreadable" (baseline) | 0.29 |
| Shape features + kNN | 0.29 |
| HOG + SVM | 0.36 |
| Small CNN (from scratch) | 0.40 |
| **CNN + visibility filter** | **0.42** |
| End-to-end CNN (from scratch) | 0.40 (0.68 on validation) |

When the reader is confident (τ = 0.8), 67% of its answers are correct, and 93% of players without a visible number are correctly called "unreadable".

![Ablation](figures/jersey/ablation.png)

## Main sources of error

Strong shadows outside the grass hue range, overlapping and very small players, pale numbers on white kits, and blur that fuses two digits into one blob. Deblurring and other clean-up steps did not recover these: the detail is missing from the image.

## Repository layout

```
notebooks/
  01_field_and_teams.ipynb        field, players and teams (Muzammil)
  02_jersey_numbers.ipynb         jersey-number pipeline, experiments A1–A4, B1–B4 (Sidra)
  03_jersey_end_to_end_cnn.ipynb  end-to-end CNN trained from scratch (Sidra; outputs not saved)
figures/                          key figures exported from the notebooks
presentation/                     progress presentation slides
```

## Running

The notebooks were written for Google Colab (a GPU helps only for the CNN). Open one in Colab, run the setup cells to download the data, then run top to bottom. Local use: `pip install -r requirements.txt`.

## Next steps

Run both halves together on the same frames, make the grass mask robust to shadows, split touching players and digits, and collect more real training digits.

## License

[MIT](LICENSE)
