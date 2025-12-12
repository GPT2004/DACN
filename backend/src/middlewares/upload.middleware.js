const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { errorResponse } = require('../utils/response');
const { FILE_UPLOAD } = require('../config/constants');

// Ensure upload directories exist
const uploadDirs = {
  avatars: path.join(__dirname, '../../uploads/avatars'),
  medicalRecords: path.join(__dirname, '../../uploads/medical-records'),
  labResults: path.join(__dirname, '../../uploads/lab-results'),
};

Object.values(uploadDirs).forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    let uploadPath = uploadDirs.medicalRecords;

    if (req.baseUrl && req.baseUrl.includes('avatar')) {
      uploadPath = uploadDirs.avatars;
    } else if (req.baseUrl && req.baseUrl.includes('lab')) {
      uploadPath = uploadDirs.labResults;
    }

    cb(null, uploadPath);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, file.fieldname + '-' + uniqueSuffix + ext);
  }
});

// File filter
const fileFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase().substring(1);
  
  if (FILE_UPLOAD.ALLOWED_TYPES.includes(ext)) {
    cb(null, true);
  } else {
    cb(new Error(`File type .${ext} is not allowed. Allowed types: ${FILE_UPLOAD.ALLOWED_TYPES.join(', ')}`), false);
  }
};

// Upload middleware
const upload = multer({
  storage: storage,
  limits: {
    fileSize: FILE_UPLOAD.MAX_SIZE,
  },
  fileFilter: fileFilter,
});

// Error handler for multer
const handleUploadError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return errorResponse(res, `File too large. Max size: ${FILE_UPLOAD.MAX_SIZE / 1024 / 1024}MB`, 400);
    }
    return errorResponse(res, err.message, 400);
  } else if (err) {
    return errorResponse(res, err.message, 400);
  }
  next();
};

module.exports = {
  upload,
  handleUploadError,
  uploadSingle: (fieldName) => [upload.single(fieldName), handleUploadError],
  uploadMultiple: (fieldName, maxCount = 5) => [upload.array(fieldName, maxCount), handleUploadError],
};