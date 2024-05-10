const { Router } = require("express");
const { signin, signup, userDetails } = require("../controllers/users");
const router = Router();

router.post("/signup", signup);
router.post("/signin", signin);
router.post("/getUser", userDetails);

module.exports = router;
