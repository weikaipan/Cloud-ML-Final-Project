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
            let sent = "Positive! Have a gread day!";
            if (res.sentiment < 0.5) {
                sent = "Uh oh! Negative, Good luck!";
            }
            $('#sentiment').text(sent);
            $('#score').text(res.sentiment);
        });

        ajax.fail(function(res){
            document.getElementById('score-loader').style.display = "none";
            document.getElementById('text-loader').style.display = "none";
            console.log("Failed");
        });
    })

    $("#train-btn").click(function () {
        // var id = $("#model").val();
        // console.log(id)
        // console.log("print training model id")

        // var urlString = "/train";
        // var data = {
        //     "id": parseInt(id,10),
        // };

        $.ajax({
            type: 'POST',
            url: '/train',
            success: function(data, status, request) {
                status_url = request.getResponseHeader('Location');
                console.log(status_url);
                $('#task-id').val(status_url);
                fetchLogs(status_url);
            },
            error: function() {
                alert('Unexpected error');
            }
        });
    });

    $('#refresh-btn').click(function () {
        $.ajax({
            type: 'GET',
            url: $('#task-id').val(),
            success: function (data) {
                console.log(data);
            },
            error: function () {
                alert('Unexpected error');
            }
        });
    });

    function fetchLogs(status_url) {
        // send GET request to status URL
        $.getJSON(status_url, function(data) {
            // update UI
            if (data.Completed) {
                $('#termynal').append('<span data-ty>Your task is completed!</span>');
            } else if (data.verbose) {
                if (data.state === 'PENDING') {
                    $('#termynal').append('<span data-ty>Your task is pending...</span>');                    
                }
                if (data.reading && $('.read-data') === null) {
                    $('#termynal').append('<span class="read-data" data-ty>' + data.reading + '</span>');
                }
                if (data.epoch) {
                    $('#termynal').append('<span data-ty>' + data.epoch + '</span>');
                }
                if (data.train) {
                    $('#termynal').append('<span data-ty>' + data.train + '</span>');
                }
                if (data.val) {
                    $('#termynal').append('<span data-ty>' + data.val + '</span>');
                }
                if (data.test) {
                    $('#termynal').append('<span data-ty>' + data.test + '</span>');
                }
                setTimeout(function() {
                    fetchLogs(status_url);
                }, 2000);
            } else {
                $('#termynal').append('<p>Oops! Something wrong with our server, sorry!</p>');
            }
        });
    }

    // ****************************************
    // Clear the form
    // ****************************************

    $("#clear-btn").click(function () {
        $("#inventory_id").val("");
        clear_form_data()
    });


    // ****************************************
    // Search for a inventory
    // ****************************************

    $("#search-btn").click(function () {

        var urlString = "/logs";

        var ajax = $.ajax({
            type: "GET",
            url: urlString
        });

        ajax.done(function(res){
            $("#search_results").empty();
            $("#search_results").append('<div>');
            $("#search_results").append('<text>' + res + "</text>");
            $("#search_results").append('</div>');
        });

        ajax.fail(function(res){
        });

    });

})
