# Test-split results

2572 bookings from 2025-11-01 onward.

## Overall (test split, 2572 bookings)

```
             value
Mean k of 4  34.9%
All 4 match   4.1%
```

## Per-output metrics

```
         accuracy macro_f1 top3_accuracy within_1_day within_1_hour mae_hours day_accuracy clock_within_30min clock_within_60min clock_mae_min
facility    45.6%    40.0%         81.6%                                                                                                      
day         40.1%                               60.4%                                                                                         
hour        38.3%                                             57.7%      3.51                                                                 
nudge       15.4%                                                                    22.9%              63.1%              64.8%        172.08
```

## Model vs baselines

```
                          Facility acc Day acc Hour acc Nudge acc Mean k of 4 All 4 match
model (gradient boosting)        45.6%   40.1%    38.3%     15.4%       34.9%        4.1%
baseline: majority               27.8%   14.4%    18.4%      1.4%       15.5%        0.1%
baseline: repeat_last            37.6%   12.9%    27.4%      5.6%       20.9%        0.5%
baseline: resident_habit         48.8%   27.1%    40.9%     11.5%       32.1%        3.3%
```

## By amount of resident history

```
            n facility    day   hour  nudge mean_k_of_4
history                                                
1-9        52    42.3%  36.5%  34.6%  17.3%       32.7%
10-29     415    41.9%  33.7%  35.9%  10.6%       30.5%
30+      2105    46.5%  41.5%  38.8%  16.3%       35.8%
```

## By actual facility

```
                     n facility    day   hour  nudge
y_fac                                               
Gym                714    71.1%  42.4%  52.2%  16.7%
Pool               409    50.9%  37.4%  44.0%  14.9%
Badminton Court    503    44.1%  37.4%  39.2%  13.5%
Tennis Court       362    31.5%  44.8%  33.4%  14.4%
Clubhouse          316    18.7%  38.0%  18.4%  14.9%
Multipurpose Hall  268    23.5%  39.6%  20.5%  18.7%
```

## Facility confusion (counts)

```
predicted          Badminton Court  Clubhouse  Gym  Multipurpose Hall  Pool  Tennis Court
actual                                                                                   
Badminton Court                222         24  115                 17    81            44
Clubhouse                       30         59  105                 23    60            39
Gym                             38         11  508                 39    79            39
Multipurpose Hall               29         13   74                 63    45            44
Pool                            48         17   88                 18   208            30
Tennis Court                    30         24  109                 27    58           114
```
