# Phòng khám đa khoa Management System - Frontend

## 📋 Mô tả dự án
Hệ thống quản lý phòng khám đa khoa với tích hợp AI, được xây dựng bằng ReactJS.

## 🚀 Công nghệ sử dụng
- **React 18** - UI Library
- **React Router v6** - Routing
- **Tailwind CSS** - Styling
- **Axios** - HTTP Client
- **React Hook Form** - Form Management
- **Recharts** - Data Visualization
- **Socket.io** - Real-time Communication
- **Lucide React** - Icons

## 📦 Cài đặt

### Yêu cầu hệ thống
- Node.js >= 16.x
- npm >= 8.x hoặc yarn >= 1.22.x

### Các bước cài đặt

1. Clone repository
```bash
git clone <repository-url>
cd benh-vien-nhi-dong-management-frontend
```

2. Cài đặt dependencies
```bash
npm install
# hoặc
yarn install
```

3. Cấu hình môi trường
```bash
cp .env.example .env
```
Chỉnh sửa file `.env` với thông tin backend API của bạn.

4. Khởi chạy development server
```bash
npm start
# hoặc
yarn start
```

Ứng dụng sẽ chạy tại `http://localhost:3000`

## 🏗️ Cấu trúc thư mục

```
src/
├── assets/          # Images, styles
├── components/      # React components
├── pages/           # Page components
├── services/        # API services
├── hooks/           # Custom hooks
├── context/         # React context
├── utils/           # Utilities
├── routes/          # Route definitions
└── config/          # Configuration files
```

## 🔐 Authentication & Authorization

Hệ thống hỗ trợ 4 role:
- **PATIENT** - Bệnh nhân
- **DOCTOR** - Bác sĩ
- **RECEPTIONIST** - Lễ tân
- **ADMIN** - Quản trị viên

## 🌟 Tính năng chính

### Bệnh nhân
- ✅ Đặt lịch hẹn trực tuyến
- ✅ Xem hồ sơ bệnh án
- ✅ Xem đơn thuốc
- ✅ AI Symptom Checker
- ✅ Quản lý thông tin cá nhân

### Bác sĩ
- ✅ Quản lý lịch làm việc
- ✅ Xem danh sách bệnh nhân
- ✅ Khám bệnh & ghi bệnh án
- ✅ Kê đơn thuốc
- ✅ AI Risk Prediction

### Lễ tân
- ✅ Quản lý lịch hẹn
- ✅ Đăng ký bệnh nhân mới
- ✅ Check-in bệnh nhân
- ✅ Quản lý hóa đơn

### Admin
- ✅ Quản lý người dùng
- ✅ Quản lý bác sĩ
- ✅ Quản lý thuốc & tồn kho
- ✅ Báo cáo & thống kê

## 🧪 Testing

```bash
npm test
# hoặc
yarn test
```

## 🏭 Build Production

```bash
npm run build
# hoặc
yarn build
```

Build folder sẽ được tạo tại `./build`

## 📝 Scripts

- `npm start` - Chạy development server
- `npm run build` - Build production
- `npm test` - Chạy tests
- `npm run lint` - Kiểm tra lỗi code
- `npm run lint:fix` - Tự động fix lỗi
- `npm run format` - Format code với Prettier

## 🔧 Cấu hình API

Cấu hình endpoint trong file `.env`:

```env
REACT_APP_API_URL=http://localhost:3000/api
REACT_APP_AI_SERVICE_URL=http://localhost:5000
```

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết

## 👥 Tác giả

Nguyễn Thanh Tú - Sinh viên K2022

## 📞 Liên hệ

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Note**: Đây là dự án đồ án chuyên ngành, phát triển cho mục đích học tập.