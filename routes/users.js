const { Router } = require("express");
const {
  signin,
  signup,
  userDetails,
  contact,
} = require("../controllers/users");
const router = Router();

router.post("/signup", signup);
router.post("/signin", signin);
router.post("/getUser", userDetails);
router.post("/contact", contact);

module.exports = router;
