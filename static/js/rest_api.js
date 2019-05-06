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
            $('#score').text(res.sentiment);
        });

        ajax.fail(function(res){
            document.getElementById('score-loader').style.display = "none";
            document.getElementById('text-loader').style.display = "none";
            console.log("Failed");
        });
    })

    $("#train-btn").click(function () {
        var id = $("#model").val();
        console.log(id)
        console.log("print training model id")

        var urlString = "/train";
        var data = {
            "id": parseInt(id,10),
        };

        var ajax = $.ajax({
                type: "POST",
                url: urlString,
                contentType:"application/json",
                data: JSON.stringify(data)
            });

        ajax.done(function(res){
        });

        ajax.fail(function(res){
            console.log(res);
        });
    });

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
