const express = require("express");
const database = require("./models/database");
const database2 = require("./models/database2");
const app = express();
const cors = require("cors");
app.use(express.json());


app.use(
  cors({
    origin: "http://localhost:3000", // Your frontend origin
    credentials: true, // Allow credentials (cookies)
  })
);
const EnterCorrectLabel = async (req, res) => {
  console.log(req.body);
  try {
    const newdatabase = await database.create({
      CorrectLabel: req.body.label,
      image: req.body.img, // Save path or process the file
    });
    res.status(201).json({
      status: "success, posted",
      data: newdatabase,
    });
  } catch (error) {
    res.status(500).json({ status: "error", message: error.message });
  }
};

const getData = async (req, res) => {
  try {
    const data = await database.find();

    res.status(200).json({
      data: data,
    });
  } catch (error) {
    res.status(500).json({ status: "error", message: error.message });
  }
};

const EnterCorrectLabel2 = async (req, res) => {
  console.log(req.body);
  try {
    const newdatabase = await database2.create({
      CorrectLabel: req.body.label,
      image: req.body.img, // Save path or process the file
    });
    res.status(201).json({
      status: "success, posted",
      data: newdatabase,
    });
  } catch (error) {
    res.status(500).json({ status: "error", message: error.message });
  }
};

const getData2 = async (req, res) => {
  try {
    const data = await database2.find();

    res.status(200).json({
      data: data,
    });
  } catch (error) {
    res.status(500).json({ status: "error", message: error.message });
  }
};

app.route("/").post(EnterCorrectLabel).get(getData);
app.route("/database2").post(EnterCorrectLabel2).get(getData2);

module.exports = app;