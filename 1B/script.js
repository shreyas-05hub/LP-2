$(document).ready(function() {
    $("#addStudent").click(function(event) {
        event.preventDefault(); // Prevent form submission

        let studentData = getStudentData();
        localStorageUse(studentData);
        window.location.href = "display.html";
    });

    function getStudentData() {
        let student = {
            firstName: $("#firstName").val(),
            lastName: $("#lastName").val()
        };
        return student;
    }

    function localStorageUse(studentData) {
        if (!localStorage.getItem("student")) {
            localStorage.setItem("student", JSON.stringify(studentData));
        } else {
            localStorage.removeItem("student");
            localStorage.setItem("student", JSON.stringify(studentData));
        }
    }
});
