let mode = "manual";

// MODE SWITCH
function setMode(m) {
    mode = m;

    fetch('/set_mode/' + m);

    document.getElementById("mode-status").innerText =
        "Mode: " + m.toUpperCase();
}

// INPUT FIELDS
function selectComp() {
    document.getElementById("input-fields").innerHTML =
        `<input id="Width" placeholder="Width">
         <input id="Height" placeholder="Height">`;
}

// MANUAL CHECK
function checkDimensions() {
    if (mode !== "manual") return;

    let width = parseFloat(document.getElementById("Width").value);
    let height = parseFloat(document.getElementById("Height").value);

    fetch('/check', {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            dimensions: {Width: width, Height: height}
        })
    })
    .then(res => res.json())
    .then(data => showResult(data.status));
}

// AUTO LIVE UPDATE
setInterval(() => {
    if (mode !== "auto") return;

    fetch('/result')
    .then(res => res.json())
    .then(data => {
        showResult(data.status + " (" + data.type + ")");
    });

}, 500);

// DISPLAY RESULT
function showResult(text) {
    let div = document.getElementById("result-status");

    div.innerHTML = text;

    if (text.includes("PASS"))
        div.style.color = "lime";
    else if (text.includes("FAIL"))
        div.style.color = "red";
    else
        div.style.color = "yellow";
}