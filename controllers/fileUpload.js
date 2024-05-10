
module.exports.uploadFileToCloudinary = async (req, res) => {
    await res.json(req.file);
}