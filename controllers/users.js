const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
// const jwt_decode = require("jwt-decode");
const User = require("../models/userModel");
require("dotenv").config();

module.exports.signup = async (req, res) => {
  // console.log("hi");
  const { username, email, password } = req.body;
  // console.log(name);
  try {
    const hashpw = await bcrypt.hash(password, 12);
    const user = new User({
      name: username,
      email,
      password: hashpw,
    });
    await user.save();
    const token = jwt.sign(
      {
        email: user.email,
      },
      "sickbay.ai-hackfest"
    );
    console.log(token);
    res
      .cookie("token", token, {
        httpOnly: true,
        secure: true,
        maxAge: 24 * 60 * 60 * 1000,
      })
      .json(token);
  } catch (err) {
    console.log(err);
    res.json({ msg: err });
  }
};

module.exports.userDetails = async (req, res) => {
  try {
    const token = req.cookies.token;
    const decodedToken = jwt.verify(token, process.env.TOKEN_SECRET);
    return decodedToken.id;
  } catch (error) {
    console.log(error);
  }
};

module.exports.signin = async (req, res) => {
  const { email, password } = req.body;
  const user = await User.find({
    email,
  });
  console.log(user);
  if (user.length > 0) {
    const isPasswordValid = await bcrypt.compare(password, user[0].password);
    if (isPasswordValid) {
      console.log(user);
      const token = jwt.sign(
        {
          email: user[0].email,
        },
        "sickbay.ai-hackfest"
      );
      console.log("token: ", jwt.decode(token));
      res
        .cookie("token", token, { httpOnly: true, maxAge: 8640000 })
        .json({ status: "ok" });
    } else {
      return res.json({ status: "error", user: "invalid password" });
    }
  } else {
    return res.json({ status: "error", user: "false" });
  }
};
