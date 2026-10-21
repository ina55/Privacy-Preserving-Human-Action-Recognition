# UPB transfer evaluation summary

## Overall

| Metric | Value |
| --- | ---: |
| Total UPB videos | 345 |
| Mapped videos | 345 |
| Unmapped videos | 0 |
| Mapped Top-1 Accuracy | 15.36% (53/345) |
| Mapped Top-5 Accuracy | 48.99% (169/345) |

## Per-class

| Subset | Action | UPB class | Mapped HMDB51 | Top-1 | Top-5 | Mapped | Top-1 predictions |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| DatasetHAR | A42 | staggering | walk, stand, fall_floor | 70.18% | 98.25% | 57 | walk:40, climb_stairs:11, kick_ball:3, run:1, kick:1 |
| HAR_ActiuniNoi | A1-v1 | drink water from bottle | drink | 0.00% | 9.38% | 32 | clap:15, shoot_gun:10, wave:3, pour:2, brush_hair:1 |
| HAR_ActiuniNoi | A1-v2 | drink water from cup | drink | 0.00% | 0.00% | 32 | clap:28, wave:4 |
| HAR_ActiuniNoi | A2 | eat meal | eat | 0.00% | 0.00% | 32 | clap:31, shoot_gun:1 |
| HAR_ActiuniNoi | A28 | phone call | talk | 0.00% | 3.12% | 32 | clap:22, wave:5, hug:1, walk:1, draw_sword:1 |
| HAR_ActiuniNoi | A8 | sit down | sit | 6.25% | 78.12% | 32 | walk:17, stand:6, sit:2, wave:2, fall_floor:1 |
| HAR_ActiuniNoi | A80 | squat down | sit, stand, situp | 15.62% | 65.62% | 32 | clap:10, stand:5, pick:4, wave:4, catch:2 |
| HAR_ActiuniNoi | A9 | stand up | stand | 6.25% | 93.75% | 32 | walk:23, sit:2, stand:2, climb_stairs:2, clap:1 |
| HAR_ActiuniNoi | C1 | sit down, stand up, phone call | sit, stand, talk | 0.00% | 43.75% | 32 | walk:10, wave:9, clap:5, pick:3, throw:2 |
| HAR_ActiuniNoi | C2 | stand up, drink water from cup, sit down | stand, drink, sit | 12.50% | 59.38% | 32 | throw:12, wave:5, pick:5, stand:3, shoot_gun:2 |
