# Direction Vectors: Evaluation Summary

## Overall Statistics

- Options: 200
- Dimensions: 15
- Embedding dimensionality: 4096
- Average held-out R²: 0.7396
- Average held-out Pearson r: 0.8680
- Average held-out Spearman ρ: 0.8382

## Per-Dimension Metrics


| Dim | Name                | alpha | R²_in | R²_cv | R²_held | Pearson | Spearman | cos(ridge,contrast) | cos(pre,post_orth) |
| --- | ------------------- | ----- | ----- | ----- | ------- | ------- | -------- | ------------------- | ------------------ |
| 1   | Conciseness         | 0.01  | 0.996 | 0.969 | 0.750   | 0.875   | 0.846    | 0.357               | 1.000              |
| 2   | Structure           | 0.1   | 0.910 | 0.910 | 0.578   | 0.762   | 0.769    | 0.481               | 0.998              |
| 3   | Actionability       | 0.01  | 0.997 | 0.990 | 0.795   | 0.930   | 0.901    | 0.309               | 0.852              |
| 4   | Clarity             | 0.1   | 0.929 | 0.956 | 0.613   | 0.801   | 0.847    | 0.489               | 0.743              |
| 5   | Emotional Resonance | 0.01  | 0.997 | 0.983 | 0.807   | 0.901   | 0.894    | 0.314               | 0.852              |
| 6   | Depth               | 0.01  | 0.993 | 0.953 | 0.746   | 0.873   | 0.859    | 0.287               | 0.775              |
| 7   | Descriptiveness     | 0.01  | 0.995 | 0.968 | 0.659   | 0.825   | 0.777    | 0.318               | 0.673              |
| 8   | Sustainability      | 0.01  | 0.996 | 0.969 | 0.711   | 0.850   | 0.494    | 0.311               | 0.963              |
| 9   | Community Focus     | 0.1   | 0.951 | 0.982 | 0.663   | 0.828   | 0.884    | 0.508               | 0.907              |
| 10  | Historical Context  | 0.01  | 0.995 | 0.976 | 0.672   | 0.835   | 0.803    | 0.245               | 0.886              |
| 11  | Practicality        | 0.1   | 0.975 | 0.993 | 0.877   | 0.943   | 0.923    | 0.634               | 0.766              |
| 12  | Efficiency          | 0.01  | 0.997 | 0.985 | 0.893   | 0.946   | 0.934    | 0.375               | 0.622              |
| 13  | Creativity          | 0.01  | 0.997 | 0.970 | 0.769   | 0.881   | 0.891    | 0.331               | 0.628              |
| 14  | Authenticity        | 0.1   | 0.959 | 0.982 | 0.850   | 0.922   | 0.918    | 0.551               | 0.734              |
| 15  | Formality           | 0.01  | 0.993 | 0.931 | 0.709   | 0.848   | 0.832    | 0.302               | 0.722              |


## Top/Bottom Options Per Dimension (by Predicted Score)

### Conciseness (dim 1)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| H.G. Wells.                                                  | 0.9695    | 0.9754     |
| Anne Hathaway played the lead role in 'The Devil Wears Prada | 0.9432    | 0.9303     |
| Choose an instrument, set a schedule (e.g., 30 minutes, 3 ti | 0.8334    | 0.8494     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| The script for the biographical film should prioritize the e | -0.9837   | -1.0000    |
| The essence of a small town summer is captured in the laid-b | -0.8810   | -0.9085    |
| In the heart of summer, where days stretch long and warm, a  | -0.8570   | -0.8300    |


### Structure (dim 2)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Here's a draft syllabus for an environmental economics cours | 0.8872    | 0.9407     |
| For a 7-day solo trip to New Zealand on a $2000 budget:      |           |            |


Day | 0.8002 | 0.9232 |
| The American Civil War (1861-1865) had several key events:

 | 0.7885 | 0.8135 |

**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Bike tours in Tuscany are not just about the destination, bu | -0.6256   | -0.7691    |
| Yes, I can summarize long text into a short summary. Please  | -0.5986   | -0.8281    |
| In the heart of summer, where days stretch long and warm, a  | -0.5979   | -0.6385    |


### Actionability (dim 3)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| To plan and manage your podcast production, follow these ste | 0.9586    | 1.0000     |
| A great workout routine that targets the chest and can be do | 0.8830    | 0.9114     |
| For a 7-day solo trip to New Zealand on a $2000 budget:      |           |            |


Day | 0.8594 | 0.8801 |

**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Intuition is not just a product of individual minds, but als | -0.4547   | -0.4606    |
| In the heart of summer, where days stretch long and warm, a  | -0.4001   | -0.3977    |
| Summer nights in a small town, where the sun dips into the h | -0.3999   | -0.3951    |


### Clarity (dim 4)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Scramble 2 eggs, add 1/4 cup cooked sausage or bacon, 1/4 cu | 0.7055    | 0.7347     |
| For a 7-day solo trip to New Zealand on a $2000 budget:      |           |            |


Day | 0.6807 | 0.8414 |
| Try push-ups, dumbbell presses, and chest dips using a chair | 0.6618 | 0.5750 |

**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Condensing extensive texts or narratives into compact summar | -0.8695   | -1.0000    |
| The concept of equality in relationships is often oversimpli | -0.7588   | -0.8445    |
| The script for the biographical film should prioritize the e | -0.7579   | -0.8518    |


### Emotional Resonance (dim 5)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Sophia, from the moment I met you, I knew that you were some | 0.9596    | 1.0000     |
| Sophia, life is full of unexpected twists and turns, but fro | 0.9481    | 0.9399     |
| (Verse 1)                                                    |           |            |
| Summer's haze on Main Street nights                          |           |            |
| Fireflies and                                                | 0.9229    | 0.9375     |


**Bottom 3 (lowest predicted score):**


| Option                                          | Predicted | Actual BTL |
| ----------------------------------------------- | --------- | ---------- |
| To prioritize tasks, use the Eisenhower Matrix: |           |            |


1. **Urgen | -0.6875 | -0.6673 |

| H.G. Wells. | -0.6828 | -0.6805 |
| To plan and manage your podcast production, follow these ste | -0.6686 | -0.7048 |

### Depth (dim 6)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| As I reflect on my journey, I am reminded of the pivotal mom | 0.7872    | 0.8321     |
| Here's a draft syllabus for an environmental economics cours | 0.6837    | 0.7086     |
| The distinction between objective facts and subjective inter | 0.6794    | 0.7068     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| The Beatles consisted of John Lennon, Paul McCartney, George | -0.9699   | -1.0000    |
| Yes, I can summarize long text into a short summary. Please  | -0.8448   | -0.8656    |
| Try Gelato di San Crispino or Fatamorgana in Rome for unique | -0.8142   | -0.7907    |


### Descriptiveness (dim 7)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| In a world that often values grand gestures over quiet momen | 0.8907    | 0.8892     |
| The "Aetherial Aura" hairstyle is a ethereal, otherworldly d | 0.8657    | 0.8560     |
| Sophia, from the moment I met you, I knew that you were some | 0.8443    | 0.8424     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| H.G. Wells.                                                  | -0.9816   | -1.0000    |
| The Beatles consisted of John Lennon, Paul McCartney, George | -0.9725   | -0.9738    |
| Anne Hathaway played the lead role in 'The Devil Wears Prada | -0.9209   | -0.9168    |


### Sustainability (dim 8)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| For those seeking an immersive and sustainable travel experi | 0.9504    | 1.0000     |
| Sustainability and environmental responsibility are crucial  | 0.8539    | 0.8619     |
| The syllabus for the environmental economics course should p | 0.8503    | 0.8715     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Visit Old Montreal for historic charm, try poutine and smoke | -0.1118   | -0.1000    |
| Scramble 2 eggs, add 1/4 cup cooked sausage or bacon, 1/4 cu | -0.1104   | -0.1105    |
| The names of the members of the iconic music group 'The Beat | -0.1087   | -0.0990    |


### Community Focus (dim 9)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| The local children's hospital is more than just a medical fa | 0.8098    | 1.0000     |
| In the event of a disaster, hospitality professionals in the | 0.7821    | 0.8615     |
| Pediatric cancer is a harsh reality that affects too many fa | 0.7683    | 0.8591     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Try Gelato di San Crispino or Fatamorgana in Rome for unique | -0.4083   | -0.4804    |
| Try push-ups, dumbbell presses, and chest dips using a chair | -0.3305   | -0.2845    |
| Anne Hathaway played the lead role in 'The Devil Wears Prada | -0.3263   | -0.2807    |


### Historical Context (dim 10)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| The significance of the Rosetta Stone lies not only in its r | 0.9947    | 1.0000     |
| The historical significance of Brasília can be deeply unders | 0.8879    | 0.9070     |
| The biographical film about the famous jazz musician from th | 0.8804    | 0.8918     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Yes, I can summarize long text into a short summary. Please  | -0.2802   | -0.2867    |
| Choose an instrument, set a schedule (e.g., 30 minutes, 3 ti | -0.2707   | -0.2713    |
| To prioritize tasks, use the Eisenhower Matrix:              |           |            |


1. **Urgen | -0.2680 | -0.2565 |

### Practicality (dim 11)

**Top 3 (highest predicted score):**


| Option                                                  | Predicted | Actual BTL |
| ------------------------------------------------------- | --------- | ---------- |
| For a 7-day solo trip to New Zealand on a $2000 budget: |           |            |


Day | 0.7577 | 0.9114 |
| For selling B2B technology solutions to large enterprises in | 0.6908 | 0.8121 |
| Staying hydrated on a long hike can be achieved by bringing  | 0.6893 | 0.8278 |

**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| The poem 'The Road Not Taken' is a deeply personal and exist | -0.9971   | -0.9921    |
| The poem 'The Road Not Taken' is a commentary on the societa | -0.9040   | -0.9848    |
| Intuition is not just a product of individual minds, but als | -0.8775   | -1.0000    |


### Efficiency (dim 12)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| H.G. Wells.                                                  | 0.7932    | 0.7987     |
| The Beatles consisted of John Lennon, Paul McCartney, George | 0.7458    | 0.7352     |
| Scramble 2 eggs, add 1/4 cup cooked sausage or bacon, 1/4 cu | 0.7234    | 0.7149     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Intuition is not just a product of individual minds, but als | -0.9601   | -1.0000    |
| The script for the biographical film should prioritize the e | -0.8725   | -0.8757    |
| The biographical film about the famous jazz musician from th | -0.8673   | -0.8685    |


### Creativity (dim 13)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| The "Clockwork Cascade" hairstyle features a intricate, wove | 0.9623    | 1.0000     |
| Sophia, life is full of unexpected twists and turns, but fro | 0.9611    | 0.9682     |
| (Verse 1)                                                    |           |            |
| Summer's haze on Main Street nights                          |           |            |
| Fireflies and                                                | 0.9596    | 0.9839     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| The Beatles consisted of John Lennon, Paul McCartney, George | -0.8253   | -0.8356    |
| Anne Hathaway played the lead role in 'The Devil Wears Prada | -0.8207   | -0.8158    |
| For a small TV, consider the Sonos Beam or Bose Home Speaker | -0.7994   | -0.8041    |


### Authenticity (dim 14)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| To truly experience the diversity and richness of Andhra cui | 0.8833    | 1.0000     |
| In a world that often values grand gestures over quiet momen | 0.8641    | 0.9406     |
| Sophia, from the moment I met you, I knew that you were some | 0.8507    | 0.9661     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| H.G. Wells.                                                  | -0.6319   | -0.7239    |
| Yes, I can summarize long text into a short summary. Please  | -0.6309   | -0.6433    |
| Try push-ups, dumbbell presses, and chest dips using a chair | -0.6251   | -0.5212    |


### Formality (dim 15)

**Top 3 (highest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Here's a draft syllabus for an environmental economics cours | 0.7079    | 0.7300     |
| A moral dilemma is a test of one's character, requiring the  | 0.7074    | 0.7706     |
| Condensing extensive texts or narratives into compact summar | 0.6995    | 0.7440     |


**Bottom 3 (lowest predicted score):**


| Option                                                       | Predicted | Actual BTL |
| ------------------------------------------------------------ | --------- | ---------- |
| Try push-ups, dumbbell presses, and chest dips using a chair | -1.0008   | -1.0000    |
| Try Gelato di San Crispino or Fatamorgana in Rome for unique | -0.8789   | -0.8630    |
| For a small TV, consider the Sonos Beam or Bose Home Speaker | -0.8731   | -0.8679    |


