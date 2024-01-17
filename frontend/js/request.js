function chat(data){
    return service({
        method: "post",
        url: 'http://localhost:8080/chat',
        data: data
    })
}

