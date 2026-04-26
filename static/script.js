/* ===========================
   PREDICT PAGE FULL JS CODE
=========================== */

function goHome(){
location.href = "/home";
}

/* Detect Mobile Device */
function isMobile(){
return /Android|iPhone|iPad|iPod/i
.test(navigator.userAgent);
}

/* Elements */
const fileInput =
document.getElementById("fileInput");

const previewImg =
document.getElementById("previewImg");

const placeholder =
document.getElementById("placeholder");

const resultCard =
document.getElementById("resultCard");

const resultText =
document.getElementById("resultText");

const bar =
document.getElementById("bar");

/* ===========================
   OPEN CAMERA / PICK IMAGE
=========================== */

function openCamera(){

if(isMobile()){

fileInput.click();

}else{

alert(
"Camera opens only on mobile. Please choose image manually."
);

fileInput.click();

}

}

/* ===========================
   IMAGE PREVIEW
=========================== */

fileInput.addEventListener(
"change",
function(){

if(this.files.length){

const file =
this.files[0];

previewImg.src =
URL.createObjectURL(file);

previewImg.style.display =
"block";

placeholder.style.display =
"none";

resultCard.style.display =
"none";

}

}
);

/* ===========================
   UPLOAD + ANALYZE IMAGE
=========================== */

async function uploadImage(){

if(!fileInput.files.length){

alert("Please choose image first");
return;

}

const formData = new FormData();

formData.append(
"image",
fileInput.files[0]
);

/* Show Loading */
resultCard.style.display =
"block";

resultText.innerHTML =
"⏳ Analyzing crop image...";

bar.style.width = "20%";

try{

const res = await fetch(
"/predict",
{
method:"POST",
body:formData
}
);

const data =
await res.json();

/* Success */
if(data.success){

resultText.innerHTML =

"🌿 <b>" +
data.prediction +
"</b><br><br>" +

"Confidence: " +
data.confidence +
"%<br><br>" +

"💊 <b>Treatment:</b><br>" +
data.treatment;

bar.style.width =
data.confidence + "%";

}

/* Error from backend */
else{

resultText.innerHTML =
"❌ " +
(data.message ||
"Prediction failed");

bar.style.width = "0%";

}

}

/* Network / JS Error */
catch(error){

resultText.innerHTML =
"❌ Server error";

bar.style.width = "0%";

}

}