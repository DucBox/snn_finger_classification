# EDA: Ellipse Features — Major_Axis, Minor_Axis, Area

Source: `snn_finger_processed_data.parquet`  
Algorithm: `cv2.fitEllipse` on adaptive-threshold of Lanczos×5 upscaled blob.  
Units: upscaled pixels (1 upscaled px = 0.82 mm).

---

## Before Filtering (n = 455,701)

### Descriptive Statistics

| Column | mean | std | p1 | p25 | p50 | p75 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| Major_Axis | 11.37 | 1.76 | 8.06 | 10.20 | 11.15 | 12.36 | 14.55 | 16.28 | 18.31 | **128.89** |
| Minor_Axis | 14.81 | 3.31 | 11.13 | 12.99 | 14.32 | 16.04 | 19.48 | 23.22 | 28.17 | **1368.95** |
| Area | 133.99 | **210.55** | 80.57 | 106.81 | 124.81 | 150.71 | 207.59 | 261.97 | 346.97 | **138,575.66** |

> `std(Area) = 210 > mean(Area) = 134` — std bị kéo hoàn toàn bởi outlier.

### Extreme Outliers

| P | Finger | Hand | Task | Major_Axis | Minor_Axis | Area |
|---|---|---|---|---|---|---|
| P20 | index | right | DRAG | 128.9 | 1368.9 | **138,575.7** |
| P1 | ring | left | TAP | 45.1 | 405.9 | **14,383.4** |
| P17 | thumb | left | TAP | 70.5 | 102.1 | 5,655.1 |
| P17 | thumb | left | TAP | 75.9 | 84.2 | 5,021.2 |
| P17 | thumb | left | TAP | 74.3 | 85.0 | 4,959.5 |
| P17 | thumb | left | TAP | 70.0 | 88.9 | 4,888.1 |
| P17 | thumb | left | TAP | 59.8 | 91.1 | 4,275.6 |
| P17 | thumb | left | TAP | 53.1 | 79.5 | 3,315.4 |
| P13 | little | right | SCROLL | 35.4 | 109.1 | 3,035.9 |
| P18 | index | right | DRAG | 26.2 | 96.3 | 1,982.2 |

### Impact on MinMaxScaler

MinMaxScaler fits to `[min, max]` = `[~80, 138,575]`:
- Median Area = 124 → scaled = `(124 − 80) / 138,495 ≈ **0.0003**`
- 99.9% của dữ liệu bị nén vào dải `[0, 0.0025]`
- **Area feature gần như vô dụng khi train**

---

## Filter Applied

**Criterion**: `Area <= 1000 px²`  
Physical rationale: p99.9 = 347 px², p99 = 262 px². Giá trị > 1000 là degenerate ellipse fit (blob quá bé hoặc frame nhiễu).

| | Before | After | Removed |
|---|---|---|---|
| **Row count** | 455,701 | **455,680** | **21 rows (0.005%)** |

---

## After Filtering (n = 455,680)

### Descriptive Statistics

| Column | mean | std | p1 | p25 | p50 | p75 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| Major_Axis | 11.37 | **1.73** | 8.06 | 10.20 | 11.15 | 12.36 | 14.54 | 16.27 | 18.28 | **27.62** |
| Minor_Axis | 14.80 | **2.52** | 11.13 | 12.99 | 14.32 | 16.04 | 19.48 | 23.20 | 28.05 | **48.72** |
| Area | 133.55 | **38.24** | 80.57 | 106.81 | 124.80 | 150.71 | 207.55 | 261.81 | 345.71 | **765.85** |

### Before vs After — Key Differences

| Metric | Before | After | Change |
|---|---|---|---|
| `Area` std | 210.55 | **38.24** | −83% ✅ |
| `Area` max | 138,575 | **765** | −99.4% ✅ |
| `Minor_Axis` max | 1368.9 | **48.7** | −96.4% ✅ |
| `Major_Axis` max | 128.9 | **27.6** | −78.6% ✅ |
| `Area` mean | 133.99 | 133.55 | ≈0 (stable) ✅ |
| `Area` p50 | 124.81 | 124.80 | ≈0 (stable) ✅ |
| Row count | 455,701 | 455,680 | −21 (−0.005%) ✅ |

### MinMaxScaler After Filter

MinMaxScaler fits to `[~80, 766]`:
- Median Area = 124 → scaled = `(124 − 80) / 686 ≈ **0.064**`
- p95 = 207 → scaled = `(207 − 80) / 686 ≈ **0.185**`
- Range `[0, 1]` được dùng đầy đủ hơn nhiều ✅

---

## Kết luận

- **21 rows** có Area > 1000 px² là degenerate ellipse fit, không phải dữ liệu hợp lệ.
- Sau khi loại: std giảm 83%, max giảm 99.4%, mean/median gần như không đổi.
- `Area` feature giờ có ý nghĩa thực sự với MinMaxScaler.
- `Major_Axis` và `Minor_Axis` cũng clean — max 27.6 và 48.7 px, hợp lý với sensor 15×27 upscaled ×5.
