# Test-split results

2736 bookings from 2025-11-01 onward.

## Overall (test split, 2736 bookings)

```
             Value
Mean k of 4  36.1%
All 4 match   3.7%
```

## Per-output metrics

```
                              Value
Output   Metric                    
facility accuracy             45.1%
         macro_f1             41.6%
         top3_accuracy        82.0%
day      accuracy             42.5%
         within_1_day         63.1%
hour     accuracy             39.6%
         within_1_hour        58.4%
         mae_hours             3.49
nudge    accuracy             17.4%
         day_accuracy         27.5%
         clock_within_30min   62.9%
         clock_within_60min   66.2%
         clock_mae_min       121.47
```

## Model vs baselines

```
                          Facility acc Day acc Hour acc Nudge acc Mean k of 4 All 4 match
Method                                                                                   
model (gradient boosting)        45.1%   42.5%    39.6%     17.4%       36.1%        3.7%
baseline: majority               26.4%   15.1%    17.0%      0.7%       14.8%        0.0%
baseline: repeat_last            37.6%   12.5%    26.7%      6.3%       20.8%        0.5%
baseline: resident_habit         47.6%   27.5%    39.8%     12.7%       31.9%        2.9%
```

## By amount of resident history (hit rates)

```
               Bookings Facility    Day   Hour  Nudge Mean k of 4
Prior bookings                                                   
1-9                  56    44.6%  30.4%  39.3%  10.7%       31.2%
10-29               423    44.9%  35.0%  37.8%  16.5%       33.6%
30+                2257    45.1%  44.2%  39.9%  17.7%       36.7%
```

## By actual facility (hit rates)

```
                  Bookings Facility    Day   Hour  Nudge
Actual facility                                         
Gym                    723    62.7%  44.3%  54.9%  17.6%
Pool                   522    51.5%  42.3%  45.8%  17.6%
Badminton Court        454    44.1%  42.5%  33.9%  17.2%
Tennis Court           374    28.6%  37.7%  33.7%  15.5%
Clubhouse              369    36.9%  42.5%  30.6%  19.0%
Multipurpose Hall      294    23.1%  44.2%  18.4%  17.0%
```

## Facility confusion (counts)

```
predicted         Badminton Court Clubhouse  Gym Multipurpose Hall Pool Tennis Court
actual                                                                              
Badminton Court               200        47  115                12   44           36
Clubhouse                      52       136   69                15   58           39
Gym                            97        32  453                37   67           37
Multipurpose Hall              32        24   96                68   46           28
Pool                           58        42  104                17  269           32
Tennis Court                   62        32  101                16   56          107
```
