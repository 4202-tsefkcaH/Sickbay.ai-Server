const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const Report = new Schema(
  {
    user: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    link: {
      type: String,
      required: true,
    },
    condition: {
      type: String,
      required: true,
    },
    cure: {
      type: String,
      required: true,
    },
  },
  { collection: "report-data" },
  { timestamps: true }
);

module.exports = mongoose.model("Report", Report);
