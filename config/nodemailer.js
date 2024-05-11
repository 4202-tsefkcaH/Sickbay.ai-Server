const nodemailer = require("nodemailer");
// import nodemailer from "./nodemailer.mjs";

module.exports.transporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  //   port: 587,
  //   secure: false, // Use `true` for port 465, `false` for all other ports
  port: 465,
  secure: true, // use SSL
  auth: {
    user: "cs.ratul03@gmail.com",
    pass: "muncvnolmxmsmwsz",
  },
});
module.exports.mailOptions = {
  from: "cs.ratul03@gmail.com",
  to: "cs.ratul03@gmail.com",
};
