const express = require('express');  // imported express module
const app = express();  //calling express function

// providing port number
app.use(express.static('public'))  // path for index.html 
app.listen(4000,()=>{   // port num, call back function and arrow function
    console.log("Server is started");  // msg shown when server is started
})