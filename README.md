# Rain Forecast (Radar + ERA5) — Radar Filling + U-Net Attention

Link dataset: https://www.kaggle.com/datasets/vanhuyluan/data-rain-ai

Dự án này gồm **2 notebook**:

1) **Tiền xử lý Radar (fill vùng trắng/NaN)** để tạo tập radar “đã làm sạch”
2) **Huấn luyện + dự báo mưa (nowcasting)** từ chuỗi radar (và ERA5) bằng mô hình **U-Net + Attention + Temporal Attention**

## 1) Files trong repo

- Notebook 1 (tiền xử lý radar): `fill-data-radar.ipynb`
- Notebook 2 (train/infer): `rain-forecast-unet-main.ipynb`

## 2) Luồng xử lý end-to-end

### 2.1 Radar filling (Notebook 1)

`fill-data-radar.ipynb` tập trung vào việc **fill các pixel bị khuyết** trong ảnh radar (thường là `nodata`, `NaN`, `-9999`, `-inf`).

Các cách fill có trong notebook:

- **Gaussian masked fill**: làm mượt bằng Gaussian nhưng chỉ fill vùng NaN gần vùng “có mưa” (theo threshold)
- **Morphological fill**: dilate vùng “có mưa”, fill holes, rồi nội suy NaN theo `nanmean` lân cận
- (Tuỳ chọn) **Kalman filter theo thời gian**: áp dụng theo chuỗi ảnh (stack) để làm mượt biến thiên theo giờ

Đầu ra thường là các file `.tif` đã fill (ví dụ hậu tố `_filled.tif`, hoặc lưu sang thư mục processed) để dùng cho notebook huấn luyện.

### 2.2 Train + Forecast (Notebook 2)

`rain-forecast-unet-main.ipynb` thực hiện:

- Load chuỗi Radar theo giờ (mặc định `seq_len = 6`)
- Load ERA5 theo thời gian (8 biến: `CAPE`, `U250`, `R250`, `TCWV`, `TCLW`, `TCW`, `V850`, `U850`)
- Tạo dataset theo mốc thời gian → train/val/test
- Huấn luyện mô hình U-Net + Attention
- Dự báo nhiều bước (sliding / autoregressive) + đánh giá + vẽ ảnh

## 3) Dữ liệu & cấu trúc thư mục kỳ vọng

### 3.1 Radar (GeoTIFF)

- Dạng file: `.tif` (GeoTIFF)
- Notebook 2 kỳ vọng quy ước tên file:
  - `Radar_YYYYMMDDHH0000.tif` (ví dụ: `Radar_20190401000000.tif`)
- Cấu trúc thư mục theo năm/tháng/ngày:
  - `radar_root/YYYY/MM/DD/Radar_YYYYMMDDHH0000.tif`

### 3.2 ERA5 (GeoTIFF theo biến)

- Mỗi biến là một thư mục con: `era5_root/<VAR>/YYYY/MM/DD/*.tif`
- Notebook 2 hiện đang trích đặc trưng bằng **`nanmean` toàn ảnh** cho mỗi biến theo timestamp (nhanh, nhẹ, nhưng mất thông tin không gian)

## 4) Mô hình & huấn luyện (Notebook 2)

### 4.1 Input/Output

- Input radar: chuỗi `[B, T, 1, H, W]` (mặc định `T=6`)
- Input meteo: chuỗi `[B, T, M]` (mặc định `M=8`)
- Output: ảnh mục tiêu `[B, 1, H, W]` tại `forecast_horizon` (mặc định `1`)

### 4.2 Kiến trúc

- **Temporal attention** để cân trọng số theo chuỗi frame radar
- **Attention U-Net**: attention ở skip-connection
- **Fusion ERA5**: trung bình theo thời gian rồi chiếu tuyến tính và cộng residual vào bottleneck/decoder

### 4.3 Loss / imbalance handling / training tricks

- `HybridLoss = 0.7 * MSE + 0.3 * DiceLoss` (ngưỡng mưa mặc định `0.1`)
- `WeightedRandomSampler` để giảm lệch lớp (ít mưa / nhiều mưa)
- `AdamW`, scheduler (OneCycleLR), early stopping, gradient clipping

### 4.4 Metric

- RMSE, Bias, Pearson R, CSI (ngưỡng `0.1`)

## 5) Cách chạy

### 5.1 Dependencies

Khuyến nghị dùng môi trường riêng. Cài tối thiểu:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch rasterio scikit-image
```

Gợi ý Windows (nếu `rasterio` lỗi do GDAL):

```bash
conda create -n rain-forecast python=3.10 -y
conda activate rain-forecast
conda install -c conda-forge rasterio -y
pip install numpy pandas scipy scikit-learn matplotlib torch scikit-image
```

### 5.2 Thứ tự chạy notebook

1. Chạy `fill-data-radar.ipynb` để tạo radar đã fill (làm sạch vùng trắng/NaN)
2. Chạy `rain-forecast-unet-main.ipynb` để train + forecast

### 5.3 Lưu ý về đường dẫn (Kaggle vs Local)

Cả hai notebook hiện có nhiều đường dẫn kiểu Kaggle (ví dụ `/kaggle/input/...`, `/kaggle/working/...`). Khi chạy local, bạn cần sửa các biến đường dẫn ở các cell cuối notebook (phần chạy pipeline) để trỏ tới dữ liệu trên máy.

## 6) Ghi chú

- Repo hiện ở dạng notebook/PoC; chưa có module hoá (`src/`), config file, hay lưu checkpoint chính thức.
- Nếu bạn muốn, mình có thể giúp tách thành code chuẩn (train/infer scripts) hoặc thêm phần lưu model + log.
