$(function () {

    $("#sentence-btn").click(function () {
        console.log(document.getElementById('score-loader'));
        document.getElementById('score-loader').style.display = "block";
        document.getElementById('text-loader').style.display = "block";

        let sentence = $('#sentence').val();

        let ajax = $.get({
            url: "/test",
            data: {
                "sentence": sentence,
            }
        });

        ajax.done(function (res) {
            console.log(res);
            document.getElementById('score-loader').style.display = "none";
            document.getElementById('text-loader').style.display = "none";
            let sent = "Positive! Have a gread day! 😀";
            if (res.sentiment < 0.5) {
                sent = "Uh oh! Negative, Good luck! 😭";
            }
            $('#sentiment').text(sent);
            $('#score').text(res.sentiment);
        });

        ajax.fail(function (res){
            document.getElementById('score-loader').style.display = "none";
            document.getElementById('text-loader').style.display = "none";
            console.log("Failed");
        });
    })

    $("#train-btn").click(function () {
        let embedding = parseInt($('#embedding').val()),
            lr = parseFloat($('#lr').val()),
            optim = $('#optim').val(),
            model = $('#model').val();
            batch = parseInt($('#batch').val());
            epoch = parseInt($('#epoch').val());
        let data = {
            'embedding': embedding,
            'lr': lr,
            'optim': optim,
            'model': model,
            'batch': batch,
            'epoch': epoch
        };

        console.log(data)

        $.ajax({
            type: 'POST',
            url: '/train',
            data: JSON.stringify(data),
            contentType: 'application/json',
            success: function(data, status, request) {
                const status_url = request.getResponseHeader('Location');
                const task_id = request.getResponseHeader('task_id');
                console.log(status_url);
                fetchLogs(status_url);
            },
            error: function() {
                alert('Unexpected error');
            }
        });

        function fetchLogs(status_url) {
            // send GET request to status URL
            $.getJSON(status_url, function(data) {
                // update UI
        
                if (data.error) {
                    $('#termynal').prepend('<p>Oops! Something wrong with our server, sorry!</p>');    
                } else {
                    if (data.state === 'PENDING') {
                        if (document.querySelector('#pending') === null) {
                            $('#termynal').prepend('<span data-ty id="pending">Your task is pending...</span>');                    
                        }
                        setTimeout(function() {
                            fetchLogs(status_url);
                        }, 2000);
                    } else {
                        if (data.state === 'READING') {
                            if (document.querySelector('#read-data') === null) {
                                let configs = '<span data-ty id="read-data">' + data.task.INFO + '</span>';
                                configs += 'Parameters Settings';
                                configs += '<span data-ty>================================</span>';
                                for (let key in data.task) {
                                    if (key !== 'INFO') {
                                        configs += '<span data-ty>' + key + '=' + data.task[key] + '</span>';
                                    }
                                }
                                configs += '<span data-ty>================================</span>';
                                $('#termynal').prepend(configs);
                            }
                            setTimeout(function() {
                                fetchLogs(status_url);
                            }, 2000);
                        } else if (data.state === 'START') {
                            if (document.querySelector('#start') === null) {
                                $('#termynal').prepend('<span data-ty id="start">START TRAINING</span>');
                            }
                            setTimeout(function() {
                                fetchLogs(status_url);
                            }, 2000);
                        } else if (data.state === 'END' || data.state === 'PROGRESS') {
                            $('#termynal').prepend('<span data-ty>' + data.task.Epoch + '</span>');
                            $('#termynal').prepend('<span data-ty>' + data.task.Train + '</span>');
                            $('#termynal').prepend('<span data-ty>' + data.task.Val + '</span>');
        
                            if (data.state === 'END') {
                                $('#termynal').prepend('<span data-ty>Your task is completed!</span>');
                                document.getElementById('form_data').style.display = 'none';
                                document.getElementById('re-train').style.display = 'inline-block';
                                document.getElementById('re-train-link').style.display = 'inline-block';
                            } else {
                                // PROGRESS, continue fetching log
                                setTimeout(function() {
                                    fetchLogs(status_url);
                                }, 2000);
                            }
                        } else if (data.state === 'SUCCESS') {
                            if (data.task.Completed) {
                                $('#termynal').prepend('<span data-ty>' + data.task.Epoch + '</span>');
                                $('#termynal').prepend('<span data-ty>' + data.task.Test + '</span>');
                            } else {
                                $('#termynal').prepend('<span data-ty>' + data.task.Epoch + '</span>');
                                $('#termynal').prepend('<span data-ty>' + data.task.Train + '</span>');
                                $('#termynal').prepend('<span data-ty>' + data.task.Val + '</span>');
                            }
                            document.getElementById('form_data').style.display = 'none';
                            document.getElementById('re-train').style.display = 'inline-block';
                            document.getElementById('re-train-link').style.display = 'inline-block';
                        }
                    }
                }
            });
        }
        
    });

})
