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

    // ****************************************
    // Create a inventory
    // ****************************************

    $("#create-btn").click(function () {

        var id = $("#inventory_id").val();
        var count = $("#inventory_count").val();
        var condition = $("#inventory_condition").val();
        var inventory_reorder_point = $("#inventory_reorder_point").val();
        var inventory_restock_level = $("#inventory_restock_level").val();

        var data = {
            "id": parseInt(id,10),
            "condition": condition,
            "count": parseInt(count,10),
            "reorder_point": parseInt(inventory_reorder_point,10),
            "restock_level": parseInt(inventory_restock_level,10),
        };
        var ajax = $.ajax({
            type: "POST",
            url: "/api/inventory",
            contentType:"application/json",
            data: JSON.stringify(data),
        });

        ajax.done(function(res){
            update_form_data(res)
            flash_message("Success")
        });

        ajax.fail(function(res){
            flash_message(res.responseJSON.message)
        });
    });


    // ****************************************
    // Update a inventory
    // ****************************************

    $("#update-btn").click(function () {
        var id = $("#inventory_id").val();
        var count = $("#inventory_count").val();
        var condition = $("#inventory_condition").val();
        var inventory_reorder_point = $("#inventory_reorder_point").val();
        var inventory_restock_level = $("#inventory_restock_level").val();

        var data = {
            "id": parseInt(id,10),
            "condition": condition,
            "count": parseInt(count,10),
            "reorder_point": parseInt(inventory_reorder_point,10),
            "restock_level": parseInt(inventory_restock_level,10),
        };

        var ajax = $.ajax({
                type: "PUT",
                url: "/api/inventory/" + id,
                contentType:"application/json",
                data: JSON.stringify(data)
            });

        ajax.done(function(res){
            update_form_data(res)
            flash_message("Success")
        });

        ajax.fail(function(res){
            flash_message(res.responseJSON.message)
        });

    });

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

    $("#delete-btn").click(function () {

        var inventory_id = $("#inventory_id").val();

        var ajax = $.ajax({
            type: "DELETE",
            url: "/api/inventory/" + inventory_id,
            contentType:"application/json",
            data: '',
        });

        ajax.done(function(res){
            clear_form_data()
            flash_message("inventory with ID [" + res.id + "] has been Deleted!")
        });

        ajax.fail(function(res){
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
    // Reorder a inventory
    // ****************************************

    $("#reorder-btn").click(function () {
        var id = $("#inventory_id").val();

        var urlString = "/api/inventory";

        if(id == "") {
            urlString += "/reorder";
        } else {
            urlString += "/" + id + "/reorder";
        }

        // var data = {
        //     "id": parseInt(id,10),
        // };

        var ajax = $.ajax({
                type: "PUT",
                url: urlString,
                contentType:"application/json"
                // data: JSON.stringify(data)
            });

        ajax.done(function(res){
            update_form_data(res)
            flash_message("Success")
        });

        ajax.fail(function(res){
            flash_message(res.responseJSON.message)
        });

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
            flash_message(res.responseJSON.message)
        });

    });

})
