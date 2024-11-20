const mongoose = require("mongoose");

const dataSchema = new mongoose.Schema({
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

const database = mongoose.model("database", dataSchema);
module.exports = database;