const jwt = require("jsonwebtoken");
require("dotenv").config();

module.exports.validateUser = async (req, res, next) => {
  try {
    const token = req.cookies.token;
    console.log(req.cookies);
    const decodedToken = jwt.verify(token, process.env.TOKEN_SECRET);
    if (decodedToken) {
      return res.json({ msg: "User Validated!" }, { status: 200 });
    } else res.json({ msg: "Fake user" }, { status: 400 });
    next();
  } catch (error) {
    console.log(error);
    return res.json({ msg: "problem" }, { status: 400 });
  }
};
