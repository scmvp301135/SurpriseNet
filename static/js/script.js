$(function(){
    $(".video").parent().click(function () {
        console.log('click');
        console.log($(this).children(".playpause"));
        if(
            $(this).children(".video").get(0).paused){        
            $(this).children(".video").get(0).play();   
            $(this).children(".playpause").fadeOut();
        }
        else{       
            $(this).children(".video").get(0).pause();
            $(this).children(".playpause").fadeIn();
        }
    });  
});