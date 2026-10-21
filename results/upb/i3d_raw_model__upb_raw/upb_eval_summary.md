# UPB transfer evaluation summary (I3D HMDB51)

## Overall

| Metric | Value |
| --- | ---: |
| Total UPB videos | 345 |
| Mapped videos | 345 |
| Unmapped videos | 0 |
| Mapped Top-1 Accuracy | 31.30% (108/345) |
| Mapped Top-5 Accuracy | 50.14% (173/345) |

## Per-class

| Subset | Action | UPB class | Mapped HMDB51 | Top-1 | Top-5 | Mapped | Top-1 predictions |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| DatasetHAR | A42 | staggering | walk, stand, fall_floor | 91.23% | 100.00% | 57 | walk:52, draw_sword:1, hug:1, turn:1, sit:1 |
| HAR_ActiuniNoi | A1-v1 | drink water from bottle | drink | 0.00% | 6.25% | 32 | brush_hair:6, draw_sword:6, pour:5, turn:4, sword_exercise:3 |
| HAR_ActiuniNoi | A1-v2 | drink water from cup | drink | 0.00% | 0.00% | 32 | brush_hair:7, sword_exercise:7, shoot_bow:5, pour:4, walk:3 |
| HAR_ActiuniNoi | A2 | eat meal | eat | 0.00% | 3.12% | 32 | pour:11, shake_hands:9, draw_sword:8, shoot_gun:1, turn:1 |
| HAR_ActiuniNoi | A28 | phone call | talk | 0.00% | 0.00% | 32 | walk:9, draw_sword:5, turn:4, shoot_bow:4, hug:2 |
| HAR_ActiuniNoi | A8 | sit down | sit | 50.00% | 75.00% | 32 | sit:16, walk:8, stand:3, pick:2, draw_sword:1 |
| HAR_ActiuniNoi | A80 | squat down | sit, stand, situp | 59.38% | 87.50% | 32 | sit:12, stand:5, pushup:2, pullup:2, situp:2 |
| HAR_ActiuniNoi | A9 | stand up | stand | 15.62% | 78.12% | 32 | draw_sword:12, walk:5, stand:5, situp:4, handstand:2 |
| HAR_ActiuniNoi | C1 | sit down, stand up, phone call | sit, stand, talk | 12.50% | 43.75% | 32 | walk:18, draw_sword:3, sit:3, shake_hands:2, brush_hair:1 |
| HAR_ActiuniNoi | C2 | stand up, drink water from cup, sit down | stand, drink, sit | 37.50% | 68.75% | 32 | sit:10, draw_sword:7, brush_hair:3, shake_hands:3, stand:2 |
