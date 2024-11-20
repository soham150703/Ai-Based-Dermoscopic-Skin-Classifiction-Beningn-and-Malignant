const mongoose = require("mongoose");

const data2Schema = new mongoose.Schema({
  CorrectLabel: {
    type: String,
    require: [true, "Enter Correct Label"],
    trim: true,
  },

  image: {
    type: String,
    require: [true, "Upload image"],
  },
});

const database2 = mongoose.model("database2", data2Schema);
module.exports = database2;