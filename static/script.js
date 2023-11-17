function updateScale(value) {
    let scaleLabel = document.getElementById("scaleLabel");
    scaleLabel.value = value + "x";
}

function showSpinner() {
    var imageInput = document.getElementById("image");
    if (!imageInput.value.length) {
        return;
    }
    else {
        document.getElementById("loadingIndicator").style.display = "flex";
    }

}

function hideSpinner() {
    document.getElementById("loadingIndicator").style.display = "none";
}


// Initialize scale label
updateScale(document.getElementById("scale").value);