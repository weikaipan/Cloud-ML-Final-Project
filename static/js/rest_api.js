$(function () {

    // ****************************************
    //  U T I L I T Y   F U N C T I O N S
    // ****************************************

    // Updates the form with data from the response
    function update_form_data(res) {
        $("#inventory_id").val(res.id);
        $("#inventory_restock_level").val(res.restock_level);
        $("#inventory_reorder_point").val(res.reorder_point);
        $("#inventory_condition").val(res.condition);
        $("#inventory_count").val(res.count);
    }

    /// Clears all form fields
    function clear_form_data() {
        $("#inventory_id").val("");
        $("#inventory_restock_level").val("");
        $("#inventory_reorder_point").val("");
        $("#inventory_condition").val("");
        $("#inventory_count").val("");
    }


    // Updates the flash message area
    function flash_message(message) {
        $("#flash_message").empty();
        $("#flash_message").append(message);
    }


    $("#sentence-btn").click(function () {
        console.log(document.getElementById('score-loader'));
        document.getElementById('score-loader').style.display = "block";
        document.getElementById('text-loader').style.display = "block";
        console.log('clicked');
        console.log($('#sentence').val());
        let sentence = $('#sentence').val();
        const data = {
            "sentence": sentence,
        };
        console.log(JSON.stringify(data));

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

    // ****************************************
    // Retrieve a inventory
    // ****************************************

    $("#retrieve-btn").click(function () {

        var inventory_id = $("#inventory_id").val();

        var ajax = $.ajax({
            type: "GET",
            url: "/api/inventory/" + inventory_id,
            contentType:"application/json",
            data: ''
        });

        ajax.done(function(res){
            //alert(res.toSource())
            update_form_data(res)
            flash_message("Success")
        });

        ajax.fail(function(res){
            clear_form_data()
            flash_message(res.responseJSON.message)
        });

    });

    // ****************************************
    // Delete a inventory
    // ****************************************

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
            clear_form_data()
            flash_message("start training your model")
        });

        ajax.fail(function(res){
            console.log(res)
            flash_message("Server error!")
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

        // console.log(urlString);

        var ajax = $.ajax({
            type: "GET",
            url: urlString
        });

        ajax.done(function(res){
            $("#search_results").empty();
            $("#search_results").append('<div>');
            $("#search_results").append('<text>'+res+"</text>");
            $("#search_results").append('</div>');


            flash_message("Success")
        });

        ajax.fail(function(res){
            flash_message(res.responseJSON)
        });

    });

})
