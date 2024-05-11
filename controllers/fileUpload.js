module.exports.uploadFileToCloudinary = async (req, res) => {
  console.log(req.file);
  await res.json(req.file);
};
