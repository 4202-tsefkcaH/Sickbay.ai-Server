
module.exports.uploadFileToCloudinary = async (req, res) => {
    console.log("hi");
    await res.json(req.file);
}