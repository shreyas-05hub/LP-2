import { Component } from '@angular/core';

@Component({
  selector: 'app-showuser',
  templateUrl: './showuser.component.html',
  styleUrls: ['./showuser.component.css']
})
export class ShowuserComponent {
  user_records:any[]=[];
  data={
  name:" ",
  email:" ",
  address:" ",
  mobile:" ",
  password:" "
  }
  constructor(){
    this.user_records=JSON.parse(localStorage.getItem('users')||'{}');
  }
}
