const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const cookieParser = require("cookie-parser");
require("dotenv").config();
const userRoutes = require("./routes/users");
const uploadRoutes = require("./routes/fileUpload");
const app = express();

app.use(
  cors({
    credentials: true,
    origin: ["http://localhost:3000"],
  })
);
app.use(cookieParser());
app.use(express.json());

app.use("/api/upload/", uploadRoutes);
app.use("/api/", userRoutes);

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => {
    console.log("DB connected");
  })
  .catch((err) => {
    console.log(err);
  });

app.listen(4000, () => {
  console.log("server started at port 4000");
});
