let active = false;

document.querySelector('.scroller').addEventListener('mousedown',function(){
  active = true;
  document.querySelector('.scroller').classList.add('scrolling');
});

document.body.addEventListener('mouseup',function(){
  active = false;
  document.querySelector('.scroller').classList.remove('scrolling');
});
document.body.addEventListener('mouseleave',function(){
  active = false;
  document.querySelector('.scroller').classList.remove('scrolling');
});


document.body.addEventListener('mousemove',function(e){
  if (!active) return;
  let x = e.pageX;
  x -= document.querySelector('.wrapper').getBoundingClientRect().left;
  scrollIt(x);
});

function scrollIt(x){
    let transform = Math.max(0,(Math.min(x,document.querySelector('.wrapper').offsetWidth)));
    document.querySelector('.after').style.width = transform+"px";
    document.querySelector('.scroller').style.left = transform-25+"px";
}

scrollIt(150);

document.querySelector('.scroller').addEventListener('touchstart',function(){
  active = true;
  document.querySelector('.scroller').classList.add('scrolling');
});
document.body.addEventListener('touchend',function(){
  active = false;
  document.querySelector('.scroller').classList.remove('scrolling');
});
document.body.addEventListener('touchcancel',function(){
  active = false;
  document.querySelector('.scroller').classList.remove('scrolling');
});