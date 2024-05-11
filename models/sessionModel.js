const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const Session = new Schema(
  {
    user: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    name: {
      type: String,
      required: true,
    },
    chats: [
      {
        question: {
          type: String,
        },
        answer: {
          type: String,
        },
        timestamp: new Date(),
      },
    ],
  },
  { collection: "session-data" },
  { timestamps: true }
);

module.exports = mongoose.model("Session", Session);
