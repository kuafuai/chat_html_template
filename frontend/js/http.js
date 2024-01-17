let service = function () {
    let instance = axios.create({
        timeout: 60000,
        crossDomain: true
    })

    instance.interceptors.request.use(
        config => {
            return config;
        },
        error => {
            console.log(error)
            return Promise.reject(error)
        }
    );

    instance.interceptors.response.use(
        res => {
            const code = res.data.code || 200;
            const msg = res.data.msg || "系统错误";
            // 二进制数据则直接返回
            if (res.request.responseType ===  'blob' || res.request.responseType ===  'arraybuffer') {
              return res.data
            }
            if (code !== 200) {
              vm.$notify({
                  title: msg,
                  type: 'error'
              });
              return Promise.reject('error')
            } else {
              return res.data
            }
        },
        error => {
            console.log(111,error);
            return Promise.reject(error)
        }
    )
    return instance
}()
