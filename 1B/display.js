$(document).ready(function () {
    getData();
});

function getData() {
    let localData = localStorage.getItem("student");
    let obj = JSON.parse(localData);
    console.log(obj);
    $("#firstName").text(obj.firstName);
    $("#lastName").text(obj.lastName);
}
