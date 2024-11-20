const app = require("./App");
const mongoose = require("mongoose");

const DB =
  "mongodb+srv://sohamsurdas2004:CI5BxfLSWGgUlmwu@cluster0.drz2o.mongodb.net/skin_images?retryWrites=true&w=majority&appName=Cluster0";

mongoose.connect(DB).then((con) => {
  console.log("DB connection successful !");
});

app.listen(3001, () => {
  console.log("Linstening on Port : localhost:3001");
});