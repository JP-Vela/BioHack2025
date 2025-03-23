async function get_sessions(){
    let sessions = await window.pywebview.api.get_sessions();
    // alert('ook');
    // log(sessions)
    return sessions
}

async function showGraph(filename){
    // log('bonk')
    let data = await window.pywebview.api.get_filedata(filename);

    chart_container = document.getElementById('chart_container')
    chart_container.innerHTML = `<canvas id="myChart"></canvas>`

    canvas = document.getElementById('myChart')

    yValues = data.class
    xValues = []

    // log(yValues)

    for(let i = 0; i<yValues.length; i++){
        xValues.push(i)
    }

    log(xValues)
    drawGraph(canvas, xValues, yValues)
    // return data
}


function drawGraph(canvas, xValues, yValues){
    const ctx = canvas.getContext('2d');
    const myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: xValues,
            datasets: [{
                label: 'confidence',
                data: yValues,
                borderColor: 'blue',
                tension: 0.4,
                pointRadius:0,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Learning Confidence'
                }
            }
        }
    });
}



async function displaySessions(){
    sessionListGroup = document.getElementById('session-list-group')
    let sessions = await get_sessions()
    // log(sessions.length)

    for(let i = 0; i < sessions.length; i++){
        sessionListGroup.innerHTML += `
            <li class="list-group-item" onclick="showGraph('${sessions[i]}')">${sessions[i]}</li>
        `
    }

}

// showGraph(${sessions[i]})
function log(input){
    window.pywebview.api.log(input)
}
// log('apple')
