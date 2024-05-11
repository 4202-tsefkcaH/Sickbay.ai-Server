const { Router } = require('express');
const { uploadFileToCloudinary } = require('../controllers/fileUpload');
const router = Router();
const multer = require('multer');
const { storage } = require('../cloudinary');
const upload = multer({ storage });

router.post('/', upload.single('file'), uploadFileToCloudinary);

module.exports = router;
