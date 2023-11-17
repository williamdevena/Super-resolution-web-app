function updateScale(value) {
    let scaleLabel = document.getElementById("scaleLabel");
    scaleLabel.value = value + "x";
}

function showSpinner() {
    document.getElementById("loadingIndicator").style.display = "flex";
}

function hideSpinner() {
    document.getElementById("loadingIndicator").style.display = "none";
}

// Initialize scale label
updateScale(document.getElementById("scale").value);