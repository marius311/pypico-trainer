import camb4py
from camb4py.camb4py import try_str2bool
from tempfile import mktemp

class camb_pypico(camb4py.camb4py.camb_disk):
    """
    Modified to use transfer_filename and transfer_filename instead of
    transfer_filename(1) and transfer_filename(1)
    """
    def __init__(self,*args,**kwargs):
        super(camb_pypico_disk,self).__init__(*args,**kwargs)
        self.output_names.update({'transfer_filename':'transfer', 'transfer_matterpower':'mpk'})
        
    def _get_tmp_files(self, p):
        output_files = []
        if try_str2bool(p['get_scalar_cls']): output_files += ['scalar_output_file']
        if try_str2bool(p['get_vector_cls']): output_files += ['vector_output_file']
        if try_str2bool(p['get_tensor_cls']): output_files += ['tensor_output_file']
        if try_str2bool(p['do_lensing']): output_files += ['lensed_output_file', 'lensed_output_file']
        if try_str2bool(p['get_transfer']): output_files += ['transfer_filename', 'transfer_filename']
        
        output_files = {k:mktemp(suffix='_%s'%k) for k in output_files}
        param_file = mktemp(suffix='_param')
        
        return output_files, param_file