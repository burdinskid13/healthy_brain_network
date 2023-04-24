function varargout=run_suit(what,varargin)

%% functions
switch(what)
    
    case 'SUIT:run_normalization'  
        T1=varargin{1}; % fullpath to T1 file
        T2=varargin{2}; % fullpath to T2 file
        spm_dir=varargin{3}; % fullpath to spm toolbox: '/om2/user/maedbh/bin/spm12'
        
        addpath(genpath(spm_dir))
        spm('defaults', 'FMRI');
                
        run_suit('SUIT:isolate_segment', char(T1), char(T2))
        run_suit('SUIT:correct_cereb_mask');
        run_suit('SUIT:normalise_dartel');
        
    case 'ANAT:reslice_LPI'                  % STEP 1.2: Reslice anatomical image within LPI coordinate systems
        subj  = varargin{1}; % subjNum
        space_label = varargin{2}; 
        base_dir = varargin{3};

        % set suit dir
        SUIT_DIR = fullfile(base_dir, 'suit');

        SUIT_SUBJ_DIR = fullfile(SUIT_DIR, subj); dircheck(SUIT_SUBJ_DIR)
 
        % (1) Reslice anatomical image to set it within LPI co-ordinate frames
        source  = fullfile(SUIT_SUBJ_DIR, sprintf('%s_%s.nii', subj, space_label));
        dest    = fullfile(SUIT_SUBJ_DIR, sprintf('%s_%s.nii', subj, space_label));
        spmj_reslice_LPI(source,'name', dest);

        % (2) In the resliced image, set translation to zero
        V               = spm_vol(dest);
        dat             = spm_read_vols(V);
        V.mat(1:3,4)    = [0 0 0];
        spm_write_vol(V,dat);
        display 'Manually retrieve the location of the anterior commissure (x,y,z) before continuing'

    case 'ANAT:centre_AC'                    % STEP 1.3: Re-centre AC
        % Set origin of anatomical to anterior commissure (must provide
        % coordinates in section (4)).
        subj=varargin{1};
        space_label=varargin{2};
        base_dir = varargin{3};

        % set suit dir
        SUIT_DIR = fullfile(base_dir, 'suit');
        
        subj_idx = sscanf(subj, 'sub-%2d');

        SUIT_SUBJ_DIR = fullfile(SUIT_DIR, subj); dircheck(SUIT_SUBJ_DIR)
        
        img    = fullfile(SUIT_SUBJ_DIR, sprintf('%s_%s.nii', subj, space_label));
        V               = spm_vol(img);
        dat             = spm_read_vols(V);
        V.mat(1:3,4)    = loc_AC{subj_idx};
        spm_write_vol(V,dat);
        fprintf('AC centering done for %s', subj)

    case 'SUIT:isolate_segment'              % STEP 9.2:Segment cerebellum into grey and white matter
        T1=varargin{1};
        T2=varargin{2};

        % run isolation and segmentation routine
        fprintf('starting the isolation and segmentation for %s', T1)

        % run isolate and segment routine on T1 and T2 (if T2 is available)
        if isempty(T2)
            suit_isolate_seg({T1}, 'keeptempfiles',0); % was 'keeptempfiles',1
        else
            suit_isolate_seg({T1, T2}, 'keeptempfiles',0); % was 'keeptempfiles',1

    case 'SUIT:correct_cereb_mask'           % STEP 9.4:

        % isolate overlapping voxels using cortex grey matter mask and cerebellum grey matter mask
        spm_imcalc({sprintf('c7*.nii'), sprintf('c1*.nii')}, 'buffer_voxels.nii','(i1.*i2)')

        % mask buffer
        spm_imcalc({'buffer_voxels.nii'},'buffer_voxels.nii','i1>0')

        % remove buffer from cerebellum
        spm_imcalc({sprintf('c1*.nii'),'buffer_voxels.nii'},'cereb_prob_corr_grey.nii','i1-i2')

        % remove buffer from cortex
        spm_imcalc({sprintf('c7*.nii'),'buffer_voxels.nii'},'cortical_mask_grey_corr.nii','i1-i2')  

    case 'SUIT:normalise_dartel'             % STEP 9.5: Normalise the cerebellum into the SUIT template.
        % Normalise an individual cerebellum into the SUIT atlas template
        % Dartel normalises the tissue segmentation maps produced by suit_isolate
        % to the SUIT template
        % !! Make sure the mask has been corrected !! (see 'SUIT:correct_cereb_mask')
        job.subjND.gray      = {sprintf('*_seg1.nii')}; % was c_
        job.subjND.white     = {sprintf('*_seg2.nii')}; % was c_
        job.subjND.isolation= {'cereb_prob_corr_grey.nii'}; % cereb_prob_corr_grey
        suit_normalize_dartel(job);
       
    case 'SUIT:reslice_contrast'             % STEP 9.8: Reslice the contrast images from first-level GLM
        % Reslices the functional contrasts
        % from the first-level GLM using deformation from
        % 'suit_normalise_dartel'.
        % make sure that you reslice into 2mm^3 resolution
        subj=varargin{1};
        glm=varargin{2};
        base_dir = varargin{3};

        % set suit dir
        SUIT_DIR = fullfile(base_dir, 'suit');
        GLM_DIR = fullfile(base_dir, 'glm_firstlevel');
        
        subj = char(subj);
        glm = char(glm);

        mask = 'cereb_prob_corr_grey.nii';

        GLM_SUBJ_DIR = fullfile(GLM_DIR, glm, subj);
        SUIT_SUBJ_DIR = fullfile(SUIT_DIR, subj); dircheck(SUIT_SUBJ_DIR)
        source=dir(fullfile(GLM_SUBJ_DIR,'*.nii*')); % images to be resliced
        cd(GLM_SUBJ_DIR);
        
        job.subj.affineTr = {fullfile(SUIT_SUBJ_DIR,sprintf('Affine_%s_%s_seg1.mat',subj,space_label))}; % was Affine_c
        job.subj.flowfield= {fullfile(SUIT_SUBJ_DIR,sprintf('u_a_%s_%s_seg1.nii',subj,space_label))}; % was u_a_c
        job.subj.resample = {source.name};
        job.subj.mask     = {fullfile(SUIT_SUBJ_DIR, mask)};
        job.vox           = [3 3 3];
        suit_reslice_dartel(job);
        % move resliced images to SUIT folder
        source=fullfile(GLM_SUBJ_DIR,'*wd*'); 
        SUIT_SUBJ_DIR_FUNC = fullfile(SUIT_DIR, 'functional', glm, subj); dircheck(SUIT_SUBJ_DIR_FUNC);
        movefile(source,SUIT_SUBJ_DIR_FUNC);
        fprintf('contrast images have been resliced into suit space for %s \n\n', subj)
    
    case 'SUIT:reslice_glm'                  % STEP 9.8: Reslice the beta 4D images from first-level GLM
        % Reslices the functional betas
        % from the first-level GLM using deformation from
        % 'suit_normalise_dartel'.
        % make sure that you reslice into 2mm^3 resolution
        subj=varargin{1};
        glm=varargin{2};
        data_type=varargin{3}; % 'betas' or 'residuals' or 'r_square'
        base_dir = varargin{4};

        % set suit dir
        SUIT_DIR = fullfile(base_dir, 'suit');
        GLM_DIR = fullfile(base_dir, 'glm_firstlevel');
        
        subj = char(subj);
        glm = char(glm);
        data_type = char(data_type)

        spm('defaults', 'FMRI');

        native_mask = 'cereb_prob_corr_grey.nii';
        suit_mask = 'suit_grey'

        GLM_SUBJ_DIR = fullfile(GLM_DIR, glm, subj);
        SUIT_SUBJ_DIR = fullfile(SUIT_DIR, subj); dircheck(SUIT_SUBJ_DIR)
        fname=fullfile(GLM_SUBJ_DIR, sprintf('%s_%s_%s.nii', subj, glm, data_type)) % image(s) to be resliced
        cd(GLM_SUBJ_DIR);

        % read fname
        vol = spm_vol(fname);
        img = spm_read_vols(vol);

        if length(img)>3
            % convert 4D nifti to 3D nifti files
            spm_file_split(fname, GLM_SUBJ_DIR)

            % grab all beta niftis
            source=dir(fullfile(GLM_SUBJ_DIR,sprintf('*%s_0*',data_type))); % images to be resliced
            
            job.subj.affineTr = {fullfile(SUIT_SUBJ_DIR,sprintf('Affine_%s_%s_seg1.mat',subj,space_label))}; % was Affine_c
            job.subj.flowfield= {fullfile(SUIT_SUBJ_DIR,sprintf('u_a_%s_%s_seg1.nii',subj,space_label))}; % was u_a_c
            job.subj.resample = {source.name};
            job.subj.mask     = {fullfile(SUIT_SUBJ_DIR, native_mask)};
            job.vox           = [3 3 3];
            suit_reslice_dartel(job);

            % get resliced images
            resliced_images=dir(fullfile(GLM_SUBJ_DIR,sprintf('*wd*%s_0*',data_type))); 

            % convert 3D images to 4D image
            outpath = fullfile(GLM_SUBJ_DIR, sprintf('%s_%s_%s_%s.nii', subj, glm, data_type, suit_mask))
            spm_file_merge({resliced_images.name}, outpath)
            fprintf('convert resliced 3D images into 4D image \n\n')

            % delete 3D niftis from GLM_SUBJ_DIR
            niftis = dir(fullfile(GLM_SUBJ_DIR, sprintf('*%s_0*',data_type)))
            delete(niftis.name)
            fprintf('beta images have been resliced into suit space for %s \n\n', subj)
        else
            % grab 3D nifiti
            job.subj.affineTr = {fullfile(SUIT_SUBJ_DIR,sprintf('Affine_%s_%s_seg1.mat',subj,space_label))}; % was Affine_c
            job.subj.flowfield= {fullfile(SUIT_SUBJ_DIR,sprintf('u_a_%s_%s_seg1.nii',subj,space_label))}; % was u_a_c
            job.subj.resample = {fname};
            job.subj.mask     = {fullfile(SUIT_SUBJ_DIR, native_mask)};
            job.vox           = [3 3 3];
            suit_reslice_dartel(job);

            % get resliced images and change name
            outpath = fullfile(GLM_SUBJ_DIR, sprintf('%s_%s_%s_%s.nii', subj, glm, data_type, suit_mask))
            source_file = fullfile(GLM_SUBJ_DIR, sprintf('wd%s_%s_%s.nii', subj, glm, data_type))
            movefile(source_file, outpath);
        end

    case 'SUIT:vol2surf'                     % STEP 9.9: Make gifti files
        glm=varargin{1};
        base_dir = varargin{2};

        % set suit dir
        SUIT_DIR = fullfile(base_dir, 'suit');

        glm = char(glm);
        
        % map volume to surface and save out as gifti
        SUIT_DIR_FUNC = fullfile(SUIT_DIR, 'functional', glm);
        SUBJ_DIRS = dir(fullfile(SUIT_DIR_FUNC, '*group*'));

        for m=1:length(SUBJ_DIRS)
                
            nifti_files = dir(fullfile(SUIT_DIR_FUNC, SUBJ_DIRS(m).name, '*.nii')); 
            
            for n=1:length(nifti_files)
                
                % define gifti name
                out_name = strrep(nifti_files(n).name,'.nii','');
                out_path = fullfile(SUIT_DIR_FUNC, SUBJ_DIRS(m).name, strcat(out_name, '.gii'));
                
                % map vol 2 surf
                C=suit_map2surf(fullfile(SUIT_DIR_FUNC, SUBJ_DIRS(m).name, nifti_files(n).name),'stats','nanmean');
                % make gifti structure and save out
                g = gifti(C);
                save(g,out_path,'Base64Binary');
                fprintf('%s has been mapped from vol to surface\n\n', nifti_files(n).name)
            end 
        end


end

%% Local functions
function dircheck(dir)
if ~exist(dir,'dir');
    warning('%s doesn''t exist. Creating one now. You''re welcome! \n',dir);
    mkdir(dir);
end

