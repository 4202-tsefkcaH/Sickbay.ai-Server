const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
// const jwt_decode = require("jwt-decode");
const User = require("../models/userModel");
const { transporter, mailOptions } = require("../config/nodemailer");
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
      .json({ token: token, id: user._id });
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
  // console.log(user);
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
      console.log("token: " + user[0]._id);

      // .cookie("token", token, { httpOnly: true, maxAge: 8640000 })
      res.json({ token: token, id: user[0]._id });
    } else {
      return res.json({ status: "error", user: "invalid password" });
    }
  } else {
    return res.json({ status: "error", user: "false" });
  }
};

module.exports.contact = async (req, res) => {
  const data = req.body;
  console.log("hello");
  try {
    await transporter.sendMail({
      ...mailOptions,
      subject: data.subject,
      text: `From ${data.first} ${data.last}(${data.email}),
      ${data.message}`,
    });
    return res.status(200).json({ complete: "yes" });
  } catch (error) {
    console.log(error);
    return res.status(400).json({ complete: "no" });
  }
};
