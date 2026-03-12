import express from "express";
import type {Request, Response} from "express";
import "dotenv/config";

const app = express();
const port = process.env.PORT;

app.get("/health", function(req: Request, res: Response){
    res.status(200).json({message: "server is up and running", Running: true})
});

app.listen(port, function(){
    console.log(`Server Is Running at: [http://localhost:${port}]`);
    console.log(`Server Health Check : [http://localhost:${port}/health]`);
});
